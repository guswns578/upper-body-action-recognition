"""Frozen official VideoSwin iMiGUE weights vs existing ST-GCN++ outputs.

Official Swin backbone and SampleFrames are executed from downloaded source;
only registration decorators/imports are removed. No training or class filtering.
"""

from pathlib import Path
import ast, json, sys, time, hashlib
from typing import Optional
import numpy as np
import cv2, torch, mmcv
from torch import nn
from mmcv.transforms import BaseTransform

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/seated_references"
SRC = ROOT / "work/MM-Gesture"
OUT = BASE / "pretrained_comparison"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src/behavior_module"))
from run_mmn_pilot import LABELS
from action_adapter import ActionModel


def official_classes():
    tree = ast.parse((SRC / "swin.py").read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "mmaction.registry")
    ]
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            n.decorator_list = []
    ns = {"__name__": "official_swin"}
    exec(compile(tree, str(SRC / "swin.py"), "exec"), ns)
    loading = ast.parse((SRC / "loading.py").read_text(encoding="utf-8"))
    node = next(n for n in loading.body if isinstance(n, ast.ClassDef) and n.name == "SampleFrames")
    node.decorator_list = []
    sn = dict(np=np, BaseTransform=BaseTransform, Optional=Optional)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRC / "loading.py"), "exec"), sn)
    return ns["SwinTransformer3D"], sn["SampleFrames"]


class FrozenRGB(nn.Module):
    def __init__(self):
        super().__init__()
        Swin, self.Sampler = official_classes()
        self.backbone = Swin(arch="tiny", pretrained=None)
        self.cls_head = nn.Module()
        self.cls_head.fc_cls = nn.Linear(768, 32)
        path = ROOT / "work/pose_models/mmgesture_swin_tiny_rgb.pth"
        # Author-linked public checkpoint; metadata includes MMEngine objects.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(ck["state_dict"], strict=True)
        (OUT / "checkpoint_config.py").write_text(ck["meta"]["cfg"], encoding="utf-8")
        self.eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53])[None, :, None, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375])[None, :, None, None, None]
        self.sha = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def predict(self, frames, full_test=False):
        sampler = self.Sampler(
            clip_len=32, frame_interval=2, num_clips=4 if full_test else 1, test_mode=True
        )
        inds = sampler.transform(dict(total_frames=len(frames), start_index=0))[
            "frame_inds"
        ].reshape(-1, 32)
        h, w = frames.shape[1:3]
        if full_test:
            # Official ThreeCrop: left, right, middle (or top, bottom, middle).
            offsets = (
                [(0, 0), (2 * ((w - 224) // 2), 0), ((w - 224) // 2, 0)]
                if h == 224
                else [(0, 0), (0, 2 * ((h - 224) // 2)), (0, (h - 224) // 2)]
            )
        else:
            # Same test resize, one center spatial view for streaming diagnostic.
            offsets = [((w - 224) // 2, (h - 224) // 2)]
        probs = []
        started = time.perf_counter()
        for x, y in offsets:
            for ix in inds:
                a = frames[ix, y : y + 224, x : x + 224]
                t = torch.from_numpy(np.ascontiguousarray(a.transpose(3, 0, 1, 2)))[None].float()
                feat = self.backbone((t - self.mean) / self.std)
                if isinstance(feat, tuple):
                    feat = feat[0]
                logits = self.cls_head.fc_cls(feat.mean(dim=(2, 3, 4)))
                probs.append(logits.softmax(-1)[0].numpy())
        p = np.mean(probs, axis=0)
        assert np.isfinite(p).all() and np.isclose(p.sum(), 1, atol=1e-5)
        return p, dict(
            seconds=time.perf_counter() - started,
            sampled_indices=inds.tolist(),
            spatial_offsets=offsets,
            views=len(probs),
        )


def packed(p, labels):
    return dict(
        probabilities=p.tolist(),
        top5=[
            dict(class_id_zero_based=int(k), label_ko=labels[k], score=float(p[k]))
            for k in np.argsort(p)[::-1][:5]
        ],
    )


def main():
    torch.set_num_threads(4)
    rgb = FrozenRGB()
    skel = ActionModel(ROOT / "work/pose_models/stgcnpp_ntu120_2d_j.pth")
    ntu = json.loads(
        (ROOT / "src/behavior_module/action_labels_ko.json").read_text(encoding="utf-8")
    )
    meta = dict(
        rgb_model="MM-Gesture VideoSwin Tiny RGB iMiGUE 32",
        rgb_sha256=rgb.sha,
        strict_loaded_keys=len(rgb.state_dict()),
        labels=LABELS,
        label_mapping_source="iMiGUE Fig2; MM-Gesture processing_Skeleton.ipynb: label-1, 99->31",
        official_test="4 temporal clips x 3 spatial crops; RGB mean/std from checkpoint",
        streaming_test="same 100-frame windows as ST-GCN++; 1 temporal clip x center crop; not official 12-view test",
        class_filter=None,
        training=False,
        depth_input=False,
        device="cpu",
        limitations=[
            "Different training datasets and label sets; no causal attribution to RGB/depth alone.",
            "Three staged clips are diagnostic examples, not accuracy benchmark.",
            "Standalone official backbone adapter; no end-to-end MMAction runner parity validation.",
        ],
    )
    (OUT / "experiment.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name in ["5940726", "8189172", "6602739"]:
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(str(BASE / (name + ".mp4")))
        fps = cap.get(5)
        original_shape = (int(cap.get(4)), int(cap.get(3)))
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
        # Official multi-view complete-video output for both models.
        old = json.loads(
            (BASE / "stgcn_raw" / name / "predictions.json").read_text(encoding="utf-8")
        )
        recs = [
            json.loads(s)
            for s in (BASE / "stgcn_raw" / name / "pose/observations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        xy = np.zeros((1, len(frames), 17, 2), np.float32)
        sc = np.zeros((1, len(frames), 17), np.float32)
        for i, r in enumerate(recs):
            for person in r["people"]:
                if "pose" in person:
                    xy[0, i] = person["pose"]["keypoints_xy"]
                    sc[0, i] = person["pose"]["raw_scores"]
                    break
        p, timing = rgb.predict(frames, True)
        q, _ = skel.predict(xy, sc, original_shape, num_clips=10)
        whole = dict(
            video=name,
            fps=fps,
            frames=len(frames),
            rgb=packed(p, LABELS),
            skeleton=packed(q, ntu),
            rgb_timing=timing,
            rgb_folding_arms_rank=int(np.where(np.argsort(p)[::-1] == 16)[0][0] + 1),
            skeleton_cross_arms_rank=int(np.where(np.argsort(q)[::-1] == 95)[0][0] + 1),
        )
        (dest / "whole_video.json").write_text(
            json.dumps(whole, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            name,
            "whole",
            whole["rgb"]["top5"],
            "arms rank",
            whole["rgb_folding_arms_rank"],
            flush=True,
        )
        updates = []
        for u in old["updates"]:
            s, e = u["start_frame"], u["end_frame"]
            p, timing = rgb.predict(frames[s : e + 1])
            updates.append(
                dict(
                    start_frame=s,
                    end_frame=e,
                    start_s=s / fps,
                    end_s=e / fps,
                    rgb=packed(p, LABELS),
                    rgb_timing=timing,
                    skeleton=u,
                )
            )
            (dest / "timeline.json").write_text(
                json.dumps(updates, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(name, "timeline complete", len(updates), flush=True)


if __name__ == "__main__":
    main()
