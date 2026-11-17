from pathlib import Path
import sys, json, time, subprocess, hashlib
import cv2, numpy as np, torch, mmcv, imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/ppt_window_comparison"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src/seated_references"))
import compare_pretrained as cp

cp.OUT = OUT
torch.set_num_threads(4)
cv2.setNumThreads(2)
rgb = cp.FrozenRGB()
sk = cp.ActionModel(ROOT / "work/pose_models/stgcnpp_ntu120_2d_j.pth")
device = "cuda" if torch.cuda.is_available() else "cpu"
rgb.to(device)
rgb.mean = rgb.mean.to(device)
rgb.std = rgb.std.to(device)
print("RGB device", device, flush=True)
labels = json.loads(
    (ROOT / "src/behavior_module/action_labels_ko.json").read_text(encoding="utf-8")
)
conditions = [
    ("baseline4", 100, "4秒旧"),
    ("latest4", 100, "4秒最新"),
    ("latest2", 50, "2秒"),
    ("latest1", 25, "1秒"),
]
sources = {
    "arms": ROOT / "src/arms_generalization/8419422.mp4",
    "emphasis": ROOT / "src/emphasis_two_models/model_results/emphasis44/input_25fps.mp4",
}


@torch.inference_mode()
def rgb_indices(frames, ix):
    h, w = frames.shape[1:3]
    x = (w - 224) // 2
    y = (h - 224) // 2
    a = frames[ix, y : y + 224, x : x + 224]
    t = torch.from_numpy(np.ascontiguousarray(a.transpose(3, 0, 1, 2)))[None].float().to(device)
    feat = rgb.backbone((t - rgb.mean) / rgb.std)
    if isinstance(feat, tuple):
        feat = feat[0]
    p = rgb.cls_head.fc_cls(feat.mean((2, 3, 4))).softmax(-1)[0].cpu().numpy()
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-5)
    return p


config = dict(
    fps=25,
    stride_frames=12,
    conditions=[{"id": key, "window_frames": n} for key, n, _ in conditions],
    rgb_sampling="baseline: official center interval2; latest: 32 rounded linspace indices including both endpoints",
    skeleton_sampling="100 rounded linspace indices including endpoints, repeat existing coordinates; no interpolated missing points",
    training=False,
    smoothing=False,
    rgb_sha256=rgb.sha,
    skeleton_sha256=hashlib.sha256(
        (ROOT / "work/pose_models/stgcnpp_ntu120_2d_j.pth").read_bytes()
    ).hexdigest(),
    notes=[
        "Short-window ST-GCN++ upsampling intentionally bypasses upstream short-clip looping by first making 100 monotonic samples.",
        "All conditions have same spatial preprocessing and 0.48s stride. Baseline4 vs latest4 isolates RGB temporal sampling, latest4/2/128 compares time coverage.",
        "Predictions shown at buffer endpoint, CPU runtime reported separately. No future frames used.",
        "No time-labelled GT: cannot estimate action accuracy or transition detection delay.",
    ],
)
(OUT / "experiment.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
)
for vid, source in sources.items():
    d = OUT / vid
    d.mkdir(exist_ok=True)
    inp = d / "input_25fps.mp4"
    if not inp.exists():
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                *(["-ss", "9", "-t", "14"] if vid == "emphasis" else []),
                "-i",
                str(source),
                "-vf",
                "fps=25,scale=960:960:force_original_aspect_ratio=decrease:force_divisible_by=2",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "veryfast",
                str(inp),
            ],
            check=True,
        )
    pose = d / "pose"
    if vid == "emphasis" and not (pose / "summary.json").exists():
        pose.mkdir(exist_ok=True)
        original = ROOT / "src/emphasis_two_models/model_results/emphasis44/pose"
        lines = (original / "observations.jsonl").read_text(encoding="utf-8").splitlines()[225:575]
        (pose / "observations.jsonl").write_text("\n".join(lines), encoding="utf-8")
        (pose / "summary.json").write_text(
            (original / "summary.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
    if not (pose / "summary.json").exists():
        with (d / "pose.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src/behavior_tool/analyze.py"),
                    str(inp),
                    "--output",
                    str(pose),
                    "--action-model",
                    "none",
                    "--no-video",
                ],
                check=True,
                stdout=log,
                stderr=log,
            )
    recs = [
        json.loads(x)
        for x in (pose / "observations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ps = json.loads((pose / "summary.json").read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(inp))
    w = int(cap.get(3))
    h = int(cap.get(4))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        a = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(mmcv.imresize(a, mmcv.rescale_size((w, h), (float("inf"), 224))))
    cap.release()
    frames = np.stack(frames)
    N = len(frames)
    assert N == len(recs)
    tids = sorted({p["track_id"] for r in recs for p in r["people"] if "pose" in p})
    arrays = {}
    for tid in tids:
        xy = np.zeros((1, N, 17, 2), np.float32)
        sc = np.zeros((1, N, 17), np.float32)
        present = np.zeros(N, bool)
        for i, r in enumerate(recs):
            for p in r["people"]:
                if p["track_id"] == tid and "pose" in p:
                    xy[0, i] = p["pose"]["keypoints_xy"]
                    sc[0, i] = p["pose"]["raw_scores"]
                    present[i] = True
        arrays[tid] = (xy, sc, present)
    result = dict(
        id=vid,
        source_offset_s=9 if vid == "emphasis" else 0,
        source=str(source),
        frames=N,
        input_resolution=[w, h],
        tracks=tids,
        conditions={},
    )
    for key, n, _ in conditions:
        updates = []
        began = time.time()
        # Shared endpoints after warmup: all lengths divisible relative to end=99 modulo12.
        first = n - 1
        ends = sorted(set([first] + list(range(99, N, 12))))
        for end in ends:
            if end < first or end >= N:
                continue
            start = end - n + 1
            if any(start - 5 <= c <= end + 5 for c in ps.get("cut_frames", [])):
                continue
            rs = time.perf_counter()
            if key == "baseline4":
                ix = (
                    rgb.Sampler(clip_len=32, frame_interval=2, num_clips=1, test_mode=True)
                    .transform(dict(total_frames=n, start_index=0))["frame_inds"]
                    .ravel()
                )
                p = rgb_indices(frames[start : end + 1], ix)
            else:
                ix = np.rint(np.linspace(0, n - 1, 32)).astype(int)
                p = rgb_indices(frames[start : end + 1], ix)
            rgbtime = time.perf_counter() - rs
            skeleton = []
            six = np.rint(np.linspace(0, n - 1, 100)).astype(int)
            assert ix[-1] <= n - 1 and np.all(np.diff(ix) >= 0) and np.all(np.diff(six) >= 0)
            for tid, (xy, sc, seen) in arrays.items():
                if not seen[start] or not seen[end] or seen[start : end + 1].mean() < 0.9:
                    continue
                st = time.perf_counter()
                q, _ = sk.predict(xy[:, start + six], sc[:, start + six], (h, w))
                dt = time.perf_counter() - st
                skeleton.append(dict(track_id=tid, prediction=cp.packed(q, labels), seconds=dt))
            updates.append(
                dict(
                    start_s=start / 25,
                    end_s=end / 25,
                    rgb=cp.packed(p, cp.LABELS),
                    rgb_source_frames=(start + ix).tolist(),
                    skeleton_source_frames=(start + six).tolist(),
                    rgb_seconds=rgbtime,
                    skeleton=skeleton,
                )
            )
        result["conditions"][key] = updates
        (d / "predictions.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(vid, key, len(updates), round(time.time() - began, 1), flush=True)
    (d / "complete.json").write_text('{"complete":true}', encoding="utf-8")
print("ALL COMPLETE", flush=True)
