"""(gpu_env) MMA-52 영상 → RTMW-l 133점 좌표. 재개 가능(이미 저장된 영상은 건너뜀).

- 사람 박스: YOLOX tiny(GPU). 첫 프레임은 가장 큰 사람, 이후 이전 박스와 IoU 최대(MMA-52는 1인 영상).
- 포즈: RTMW-l 384×288, rtmlib과 같은 전처리(1.25 패딩 affine, BGR 그대로, mean/std)와 SimCC 디코딩.
  CUDA FP32(use_tf32=0): pose_env CPU 출력과 좌표 차 1e-4px 이내로 확인(2026-10-26).
- 저장: <out>/<split>/<video>.npz — xy[T,133,2] float32(원본 픽셀), score[T,133] float16, box[T,4], has_person[T].
  주석(CSV)은 건드리지 않는다. test(무작위 라벨)는 기본 제외.

    python extract_mma52_rtmw.py --splits val train [--limit 20]
"""
import argparse, json, time, hashlib, platform, threading, queue, traceback
from pathlib import Path
import numpy as np
import cv2
import torch  # CUDA/cuDNN DLL을 먼저 올려 onnxruntime CUDA EP가 쓰게 한다
import onnxruntime as ort
from rtmlib import YOLOX, RTMPose
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
from rtmlib.tools.pose_estimation.post_processings import get_simcc_maximum

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT/'work/pose_models'
SIZE = (288, 384)                        # (w, h)
# rtmlib과 같이 float64로 정규화한 뒤 float32로 바꾼다(float32로 정규화하면 좌표 0.07%가 한 칸씩 달라짐)
MEAN = np.array([123.675, 116.28, 103.53], np.float64); STD = np.array([58.395, 57.12, 57.375], np.float64)
CUDA = [('CUDAExecutionProvider', {'use_tf32': 0}), 'CPUExecutionProvider']


def session(path):
    so = ort.SessionOptions(); so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=CUDA)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)


def detect(det, frame):
    image, ratio = det.preprocess(frame); out = det.inference(image)
    d = np.asarray(out[0])[0]; lab = np.asarray(out[1]).reshape(-1)
    b = d[(lab == 0) & (d[:, 4] >= .3), :4]/ratio
    h, w = frame.shape[:2]; b[:, [0, 2]] = b[:, [0, 2]].clip(0, w-1); b[:, [1, 3]] = b[:, [1, 3]].clip(0, h-1)
    return b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]


def prepare(det, path):
    """영상 하나: 디코드 → 사람 박스 → 1.25 패딩 affine 크롭(uint8). 작업자 스레드에서 실행."""
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS)
    crops, centers, scales, boxes, has, prev = [], [], [], [], [], None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        b = detect(det, f)
        if len(b):
            prev = b[np.argmax((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))] if prev is None else b[np.argmax([iou(prev, x) for x in b])]
            has.append(True)
        else:
            has.append(False)
        if prev is None:              # 아직 사람이 한 번도 안 잡힘: 전체 화면으로 대체(결과는 has=False로 표시)
            box = np.array([0, 0, f.shape[1]-1, f.shape[0]-1], np.float32)
        else:
            box = prev
        c, s = bbox_xyxy2cs(np.asarray(box, np.float64), padding=1.25)   # rtmlib(bbox.tolist())과 같은 float64 계산
        img, s = top_down_affine(SIZE, s, c, f)
        crops.append(img); centers.append(c); scales.append(s); boxes.append(box)
    cap.release()
    return dict(fps=fps, crops=np.stack(crops) if crops else None, centers=np.array(centers), scales=np.array(scales),
                boxes=np.array(boxes, np.float32), has=np.array(has))


def infer(pose, v, batch):
    xs, ss = [], []
    name = pose.get_inputs()[0].name
    for i in range(0, len(v['crops']), batch):
        x = ((v['crops'][i:i+batch]-MEAN)/STD).transpose(0, 3, 1, 2)
        sx, sy = pose.run(None, {name: np.ascontiguousarray(x, dtype=np.float32)})
        locs, sc = get_simcc_maximum(sx, sy)
        c, s = v['centers'][i:i+batch, None], v['scales'][i:i+batch, None]
        xs.append(locs/2.0/np.array(SIZE)*s+c-s/2); ss.append(sc)
    return np.concatenate(xs), np.concatenate(ss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/MMA-52/videos'); ap.add_argument('--out', default='data/MMA-52/pose_rtmw')
    ap.add_argument('--splits', nargs='+', default=['val', 'train']); ap.add_argument('--limit', type=int)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', default='0/1', help='i/n: 영상 목록의 i번째 조각만 처리(별도 프로세스 병렬 실행용)')
    a = ap.parse_args()
    si, sn = map(int, a.shard.split('/'))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = YOLOX(str(MODELS/'yolox_tiny.onnx'), model_input_size=(416, 416), backend='onnxruntime', device='cpu')
    det.session = session(MODELS/'yolox_tiny.onnx')
    pose = session(MODELS/'rtmw_384.onnx')
    meta = dict(model='RTMW-dw-x-l cocktail14 384x288 (rtmw_384.onnx)',
                model_sha256=hashlib.sha256((MODELS/'rtmw_384.onnx').read_bytes()).hexdigest(),
                detector='YOLOX tiny 416, person score>=0.3, largest then IoU tracking', precision='CUDA FP32 (use_tf32=0)',
                preprocessing='rtmlib 0.0.14 top_down_affine padding 1.25, BGR, mean/std', onnxruntime=ort.__version__,
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                keypoints='COCO-WholeBody 133 (0-16 body, 17-22 feet, 23-90 face, 91-111 left hand, 112-132 right hand)')
    if si == 0:
        (out/'manifest.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    todo = []
    for sp in a.splits:
        (out/sp).mkdir(exist_ok=True)
        todo += [(sp, p) for p in sorted(Path(a.data, sp).glob('*.mp4')) if not (out/sp/(p.stem+'.npz')).exists()]
    todo = todo[si::sn]
    todo = todo[:a.limit] if a.limit else todo
    print(f'shard {a.shard}: {len(todo)} videos to process', flush=True)
    q = queue.Queue(maxsize=a.workers*2); it = iter(todo); lock = threading.Lock()

    def worker():
        while True:
            with lock:
                item = next(it, None)
            if item is None:
                q.put(None); return
            try:
                q.put((item, prepare(det, item[1])))
            except Exception:
                q.put((item, traceback.format_exc()))
    for _ in range(a.workers):
        threading.Thread(target=worker, daemon=True).start()
    done, ended, frames, t0, errors = 0, 0, 0, time.perf_counter(), []
    while ended < a.workers:
        got = q.get()
        if got is None:
            ended += 1; continue
        (sp, p), v = got
        if isinstance(v, str) or v['crops'] is None:
            errors.append({'video': f'{sp}/{p.name}', 'error': v if isinstance(v, str) else 'no frames'}); continue
        xy, sc = infer(pose, v, a.batch)
        np.savez_compressed(out/sp/(p.stem+'.npz'), xy=xy.astype(np.float32), score=sc.astype(np.float16),
                            box=v['boxes'], has_person=v['has'], fps=v['fps'])
        done += 1; frames += len(xy)
        if done % 50 == 0 or done == len(todo):
            el = time.perf_counter()-t0
            print(f'{done}/{len(todo)} videos, {frames} frames, {frames/el:.0f} fps, eta {(len(todo)-done)*el/done/60:.0f} min', flush=True)
    if errors:
        (out/f'errors_shard{si}_{int(time.time())}.json').write_text(json.dumps(errors, indent=1), encoding='utf-8')
    print(f'finished: {done} saved, {len(errors)} errors', flush=True)


if __name__ == '__main__':
    main()
