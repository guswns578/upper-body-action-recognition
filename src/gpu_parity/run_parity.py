"""CPU/CUDA 출력 동등성 확인용 추론 실행기 (학습 없음, 기존 결과 덮어쓰기 없음).

같은 입력(Pexels 8189172의 RTMPose 좌표와 RGB 프레임)으로 기존 두 모델을 실행해
확률을 npz로 저장한다. 환경/장치별로 한 번씩 실행한 뒤 compare_parity.py로 비교한다.

    python run_parity.py --device cpu  --out cpu_pose_env.npz   (pose_env)
    python run_parity.py --device cuda --out cuda_gpu_env.npz   (gpu_env)

기존 로더(action_adapter.ActionModel, compare_pretrained.official_classes)를 그대로 쓰고
텐서를 장치로 옮기는 부분만 추가했다. FrozenRGB.__init__은 기존 출력 폴더에 파일을 쓰므로
같은 단계를 여기서 부작용 없이 반복한다.
"""

import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import cv2, mmcv, torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/behavior_module"))
sys.path.insert(0, str(ROOT / "src/seated_references"))
from action_adapter import ActionModel
from compare_pretrained import official_classes

VIDEO = ROOT / "src/seated_references/8189172.mp4"
POSE = ROOT / "src/seated_references/stgcn_raw/8189172/pose/observations.jsonl"
TIMELINE = ROOT / "src/seated_references/pretrained_comparison/8189172/timeline.json"
MEAN = torch.tensor([123.675, 116.28, 103.53])[None, :, None, None, None]
STD = torch.tensor([58.395, 57.12, 57.375])[None, :, None, None, None]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def load_rgb(dev):
    Swin, Sampler = official_classes()
    m = nn.Module()
    m.backbone = Swin(arch="tiny", pretrained=None)
    m.cls_head = nn.Module()
    m.cls_head.fc_cls = nn.Linear(768, 32)
    ck = torch.load(
        ROOT / "work/pose_models/mmgesture_swin_tiny_rgb.pth",
        map_location="cpu",
        weights_only=False,
    )
    m.load_state_dict(ck["state_dict"], strict=True)
    return m.eval().to(dev), Sampler


@torch.inference_mode()
def rgb_predict(model, Sampler, frames, dev, full_test=False):
    """compare_pretrained.FrozenRGB.predict와 같은 샘플링·crop·정규화. 장치 이동만 추가."""
    sampler = Sampler(
        clip_len=32, frame_interval=2, num_clips=4 if full_test else 1, test_mode=True
    )
    inds = sampler.transform(dict(total_frames=len(frames), start_index=0))["frame_inds"].reshape(
        -1, 32
    )
    h, w = frames.shape[1:3]
    if full_test:
        offsets = (
            [(0, 0), (2 * ((w - 224) // 2), 0), ((w - 224) // 2, 0)]
            if h == 224
            else [(0, 0), (0, 2 * ((h - 224) // 2)), (0, (h - 224) // 2)]
        )
    else:
        offsets = [((w - 224) // 2, (h - 224) // 2)]
    mean, std = MEAN.to(dev), STD.to(dev)
    probs = []
    for x, y in offsets:
        for ix in inds:
            a = frames[ix, y : y + 224, x : x + 224]
            t = (
                torch.from_numpy(np.ascontiguousarray(a.transpose(3, 0, 1, 2)))[None]
                .float()
                .to(dev)
            )
            feat = model.backbone((t - mean) / std)
            if isinstance(feat, tuple):
                feat = feat[0]
            logits = model.cls_head.fc_cls(feat.mean(dim=(2, 3, 4)))
            probs.append(logits.softmax(-1)[0].double().cpu().numpy())
    return np.mean(probs, axis=0)


@torch.inference_mode()
def skel_predict(model, xy, sc, shape, dev, num_clips=1):
    """action_adapter.ActionModel.predict와 같은 전처리(CPU numpy) 후 장치에서 계산."""
    data, _ = model.preprocess(xy, sc, shape, num_clips)
    logits = model.cls_head(model.backbone(data.to(dev)))
    return logits.softmax(-1).mean(0).double().cpu().numpy()


def timed(fn, dev, repeats):
    fn()
    sync(dev)  # 예열
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        out.append(time.perf_counter() - t0)
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tf32", action="store_true", help="CUDA TF32 허용(기본은 엄격 FP32)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    # RGB 프레임: compare_pretrained.main과 같은 디코드·리사이즈
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(5)
    shape = (int(cap.get(4)), int(cap.get(3)))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(
            mmcv.imresize(f, mmcv.rescale_size((f.shape[1], f.shape[0]), (float("inf"), 224)))
        )
    cap.release()
    frames = np.stack(frames)

    # 좌표: compare_pretrained.main과 같은 첫 인물 선택·결측 0 채움
    recs = [json.loads(s) for s in POSE.read_text(encoding="utf-8").splitlines()]
    xy = np.zeros((1, len(frames), 17, 2), np.float32)
    sc = np.zeros((1, len(frames), 17), np.float32)
    for i, r in enumerate(recs[: len(frames)]):
        for person in r["people"]:
            if "pose" in person:
                xy[0, i] = person["pose"]["keypoints_xy"]
                sc[0, i] = person["pose"]["raw_scores"]
                break

    windows = [
        (u["start_frame"], u["end_frame"]) for u in json.loads(TIMELINE.read_text(encoding="utf-8"))
    ]

    skel = ActionModel(ROOT / "work/pose_models/stgcnpp_ntu120_2d_j.pth").to(dev)
    rgb, Sampler = load_rgb(dev)

    res = {}
    res["skel_windows"] = np.stack(
        [skel_predict(skel, xy[:, s : e + 1], sc[:, s : e + 1], shape, dev) for s, e in windows]
    )
    res["skel_whole10"] = skel_predict(skel, xy, sc, shape, dev, num_clips=10)
    res["rgb_windows"] = np.stack(
        [rgb_predict(rgb, Sampler, frames[s : e + 1], dev) for s, e in windows]
    )
    res["rgb_whole12"] = rgb_predict(rgb, Sampler, frames, dev, full_test=True)
    res["windows"] = np.array(windows)

    s0, e0 = windows[0]
    reps = 5 if dev.type == "cuda" else 3
    timing = dict(
        skel_window_s=timed(
            lambda: skel_predict(skel, xy[:, s0 : e0 + 1], sc[:, s0 : e0 + 1], shape, dev),
            dev,
            reps,
        ),
        rgb_window_s=timed(lambda: rgb_predict(rgb, Sampler, frames[s0 : e0 + 1], dev), dev, reps),
    )
    meta = dict(
        device=args.device,
        tf32=args.tf32,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        python=platform.python_version(),
        numpy=np.__version__,
        mmcv=mmcv.__version__,
        threads=torch.get_num_threads(),
        fps=fps,
        frames=len(frames),
        n_windows=len(windows),
        timing=timing,
    )
    np.savez(args.out, meta=json.dumps(meta), **res)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
