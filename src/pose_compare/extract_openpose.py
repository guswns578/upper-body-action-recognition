"""(gpu_env) OpenPose COCO-18 body 추출. 1.25배 확장 크롭 입력, 공식 기본(1 스케일, 망 높이 368).

크롭 안에서 여러 사람이 잡히면 박스 중심에 가장 가까운 사람(목/어깨 기준)을 고른다.
--device cuda|cpu. 시간 측정은 추론+후처리(PAF 조립) 포함.
"""

import argparse, sys, time
import numpy as np
import torch
from common import ROOT, HERE, CLIPS, MAP, SOURCES, read_frames, expand, save_raw

sys.path.insert(0, str(HERE / "vendor"))
from openpose.body import Body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("clips", nargs="*")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    body = Body(str(ROOT / "work/pose_models/openpose/body_pose_model.pth")).to(a.device)
    idx = MAP["openpose18"]
    tag = "openpose" if a.device == "cuda" else "openpose_cpu"
    for name, *_ in SOURCES:
        if a.clips and name not in a.clips:
            continue
        c = np.load(CLIPS / f"{name}.npz")
        boxes = c["boxes"]
        fps = float(c["fps"])
        W, H = c["size"]
        T = len(boxes)
        xy = np.full((T, 13, 2), np.nan)
        sc = np.full((T, 13), np.nan)
        ms = np.zeros(T)
        limit = T if a.device == "cuda" else min(T, 40)  # CPU는 속도 측정용 일부만
        for t, _, f in read_frames(
            c["path"].item(), float(c["start_s"]), float(c["start_s"]) + T / fps + 1e-6
        ):
            if t >= limit:
                break
            b = boxes[t]
            if np.isnan(b[0]):
                continue
            x1, y1, x2, y2 = expand(b, W, H).astype(int)
            crop = f[y1:y2, x1:x2]
            t0 = time.perf_counter()
            cand, subset = body(crop)
            if a.device == "cuda":
                torch.cuda.synchronize()
            ms[t] = (time.perf_counter() - t0) * 1e3
            if len(subset) == 0:
                continue
            cx, cy = (b[0] + b[2]) / 2 - x1, (b[1] + b[3]) / 2 - y1

            def dist(p):
                j = [int(p[k]) for k in (1, 2, 5) if p[k] >= 0]
                return (
                    np.inf
                    if not j
                    else np.hypot(cand[j, 0].mean() - cx, (cand[j, 1].mean() - cy) * 0.5)
                )

            p = min(subset, key=dist)
            for u, k in enumerate(idx):
                if p[k] >= 0:
                    q = cand[int(p[k])]
                    xy[t, u] = (q[0] + x1, q[1] + y1)
                    sc[t, u] = q[2]
        save_raw(tag, name, xy, sc, ms)
        print(
            name,
            tag,
            f"{np.median(ms[ms > 0]):.1f}ms/frame",
            "frames",
            int((ms > 0).sum()),
            flush=True,
        )


if __name__ == "__main__":
    main()
