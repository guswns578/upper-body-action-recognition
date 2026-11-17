from pathlib import Path
import sys
import json
import numpy as np, torch

R = Path(__file__).resolve().parents[1]
O = R / "src/cause_isolation"
sys.path.insert(0, str(R / "src/behavior_module"))
from action_adapter import ActionModel

torch.set_num_threads(3)
m = ActionModel(R / "work/pose_models/stgcnpp_ntu120_2d_j.pth")
names = (
    (R / "work/pyskl/tools/data/label_map/nturgbd_120.txt").read_text(encoding="utf-8").splitlines()
)
sources = {
    "rtmpose": R / "src/emphasis_two_models/model_results/emphasis44/pose/observations.jsonl",
    "rtmw": R / "src/emphasis_input_audit/full_hands/observations.jsonl",
}
records = {
    k: [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
    for k, p in sources.items()
}
rows = []
for clip in json.loads((R / "src/emphasis_clip_diagnostic/clips.json").read_text()):
    a = round(clip["start"] * 25)
    b = round(clip["end"] * 25)
    for name, recs in records.items():
        ps = [next(p["pose"] for p in recs[i]["people"] if "pose" in p) for i in range(a, b)]
        xy = np.array([p["keypoints_xy"][:17] for p in ps], np.float32)[None]
        sc = np.array([p["raw_scores"][:17] for p in ps], np.float32)[None]
        variants = {"complete": (xy, sc)}
        if name == "rtmw":
            old = [
                next(p["pose"] for p in records["rtmpose"][i]["people"] if "pose" in p)
                for i in range(a, b)
            ]
            oldxy = np.array([p["keypoints_xy"][:17] for p in old], np.float32)[None]
            oldsc = np.array([p["raw_scores"][:17] for p in old], np.float32)[None]
            variants["rtmw_xy_rtmpose_score"] = (xy, oldsc)
            variants["rtmpose_xy_rtmw_score"] = (oldxy, sc)
        if name == "rtmpose":
            for seconds in [0.5, 1, 2]:
                n = round(seconds * 25)
                variants[f"prefix_{seconds}s"] = (xy[:, :n], sc[:, :n])
                variants[f"tail_{seconds}s"] = (xy[:, -n:], sc[:, -n:])
            # Coordinate sequence is held fixed: isolate effect of score channels.
            variants["scores_one"] = (xy, np.ones_like(sc))
        for variant, (x, s) in variants.items():
            p, _ = m.predict(x, s, (540, 960), num_clips=1)
            top = np.argsort(p)[-5:][::-1]
            rows.append(
                dict(
                    clip=clip["id"],
                    estimator=name,
                    variant=variant,
                    frames=x.shape[1],
                    top5=[dict(id=int(k), label=names[k], score=float(p[k])) for k in top],
                )
            )
            if name == "rtmpose" and variant.startswith(("prefix_", "tail_")):
                rr = dict(keypoint=x.copy(), keypoint_score=s.copy(), img_shape=(540, 960))
                rr = m.official["PreNormalize2D"]()(rr)
                rr = m.official["FormatGCNInput"](num_person=2)(rr)
                with torch.inference_mode():
                    pp = (
                        m.cls_head(m.backbone(torch.from_numpy(rr["keypoint"])))
                        .softmax(-1)
                        .mean(0)
                        .numpy()
                    )
                tt = np.argsort(pp)[-5:][::-1]
                rows.append(
                    dict(
                        clip=clip["id"],
                        estimator=name,
                        variant=variant + "_native",
                        frames=x.shape[1],
                        top5=[dict(id=int(k), label=names[k], score=float(pp[k])) for k in tt],
                    )
                )
    print(clip["id"], "complete", flush=True)
(O / "user_pose_windows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
