"""(pose_env) 영상 구간별 주 인물 박스를 YOLOX tiny로 한 번만 계산해 모든 추출기가 공유한다.

주 인물 = 이전 프레임 박스와 IoU가 가장 큰 박스(첫 프레임은 가장 큰 박스). 사람이 없으면 NaN.
"""

import sys, time
import numpy as np
import onnxruntime as ort
from rtmlib import YOLOX
from common import ROOT, CLIPS, SOURCES, read_frames

MODELS = ROOT / "work/pose_models"


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def people(det, frame):
    image, ratio = det.preprocess(frame)
    out = det.inference(image)
    d = np.asarray(out[0])[0]
    lab = np.asarray(out[1]).reshape(-1)
    keep = (lab == 0) & (d[:, 4] >= 0.3)
    b = d[keep, :4] / ratio
    h, w = frame.shape[:2]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h - 1)
    return b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])]


def main():
    det = YOLOX(
        str(MODELS / "yolox_tiny.onnx"),
        model_input_size=(416, 416),
        backend="onnxruntime",
        device="cpu",
    )
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    det.session = ort.InferenceSession(
        det.onnx_model, sess_options=opt, providers=["CPUExecutionProvider"]
    )
    CLIPS.mkdir(exist_ok=True)
    only = set(sys.argv[1:])
    for name, path, s, e in SOURCES:
        if only and name not in only:
            continue
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(5)
        W, H = int(cap.get(3)), int(cap.get(4))
        cap.release()
        boxes, src_idx, prev, n_people = [], [], None, []
        t0 = time.perf_counter()
        for _, j, f in read_frames(path, s, e):
            b = people(det, f)
            n_people.append(len(b))
            if len(b) == 0:
                boxes.append([np.nan] * 4)
                src_idx.append(j)
                continue
            if prev is None:
                pick = b[np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))]
            else:
                pick = b[np.argmax([iou(prev, x) for x in b])]
            boxes.append(pick)
            prev = pick
            src_idx.append(j)
        np.savez(
            CLIPS / f"{name}.npz",
            boxes=np.array(boxes, np.float32),
            src_frame=np.array(src_idx),
            fps=fps,
            size=np.array([W, H]),
            path=str(path),
            start_s=s,
            n_people=np.array(n_people),
        )
        print(
            f"{name}: {len(boxes)} frames {W}x{H} {fps:.1f}fps, missing box {int(np.isnan(np.array(boxes)[:, 0]).sum())}, "
            f"max people {max(n_people)}, {time.perf_counter()-t0:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
