"""run_parity.py 결과 비교. 기준(ref)과 대상(test) npz, 그리고 10월 저장 Swin CPU 출력과 대조.

python compare_parity.py cpu_pose_env.npz cuda_gpu_env.npz [--json report.json]
"""

import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TIMELINE = ROOT / "src/seated_references/pretrained_comparison/8189172/timeline.json"
KEYS = ["skel_windows", "skel_whole10", "rgb_windows", "rgb_whole12"]


def stats(a, b):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    top5 = lambda p: np.argsort(-p, axis=1)[:, :5]
    return dict(
        n=len(a),
        max_abs_diff=float(np.abs(a - b).max()),
        top1_agree=int((a.argmax(1) == b.argmax(1)).sum()),
        top5_order_agree=int((top5(a) == top5(b)).all(1).sum()),
        top1_margin_min=float(np.min(np.sort(a, 1)[:, -1] - np.sort(a, 1)[:, -2])),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--json")
    args = ap.parse_args()
    ref, test = np.load(args.ref), np.load(args.test)
    assert (ref["windows"] == test["windows"]).all()
    report = dict(
        ref=json.loads(str(ref["meta"])),
        test=json.loads(str(test["meta"])),
        ref_vs_test={k: stats(ref[k], test[k]) for k in KEYS},
    )
    stored = np.array(
        [u["rgb"]["probabilities"] for u in json.loads(TIMELINE.read_text(encoding="utf-8"))]
    )
    report["stored_oct_rgb_vs_ref"] = stats(stored, ref["rgb_windows"])
    report["stored_oct_rgb_vs_test"] = stats(stored, test["rgb_windows"])
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
