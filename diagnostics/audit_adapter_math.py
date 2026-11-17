from pathlib import Path
import ast
import json
import sys
import pickle
import numpy as np, torch
from torch import nn

R = Path(__file__).resolve().parents[1]
O = R / "src/cause_isolation"
sys.path.insert(0, str(R / "src/behavior_module"))
from action_adapter import ActionModel

torch.set_num_threads(3)
model = ActionModel(R / "work/pose_models/stgcnpp_ntu120_2d_j.pth")
sources = {}
for p in (R / "src/behavior_module/vendor/pyskl").rglob("*.py"):
    rel = p.relative_to(R / "src/behavior_module/vendor/pyskl")
    q = R / "work/pyskl/pyskl" / rel
    if q.exists():
        sources[str(rel)] = p.read_bytes() == q.read_bytes()


def method(path, classname, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cl = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
    fn = next(n for n in cl.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.decorator_list = []
    ns = dict(np=np, torch=torch, F=torch.nn.functional, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), ns)
    return ns[name]


class Wrapper:
    with_cls_head = True
    test_cfg = {"average_clips": "prob"}
    backbone = model.backbone
    cls_head = model.cls_head
    average_clip = method(
        R / "work/pyskl/pyskl/models/recognizers/base.py", "BaseRecognizer", "average_clip"
    )
    forward_test = method(
        R / "work/pyskl/pyskl/models/recognizers/recognizergcn.py", "RecognizerGCN", "forward_test"
    )

    def extract_feat(self, x):
        return self.backbone(x)


ref = Wrapper()
with (O / "selected_ntu.pkl").open("rb") as f:
    sample = pickle.load(f)[0]
with torch.inference_mode():
    data, _ = model.preprocess(
        sample["keypoint"], sample["keypoint_score"], sample["img_shape"], num_clips=10
    )
    a = model.cls_head(model.backbone(data)).softmax(-1).mean(0).numpy()
    b = ref.forward_test(data[None])[0]
    diff = float(np.max(np.abs(a - b)))


class Head(nn.Module):
    forward = method(R / "work/MM-Gesture/i3d_head.py", "I3DHead", "forward")

    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(768, 32)


head = Head().eval()
torch.manual_seed(0)
feature = torch.randn(2, 768, 4, 7, 7)
with torch.inference_mode():
    diffhead = float((head(feature) - head.fc_cls(feature.mean((2, 3, 4)))).abs().max())
result = dict(
    vendored_files_match=sources,
    stgcn_official_forward_test_max_absolute_difference=diff,
    swin_official_head_max_absolute_difference=diffhead,
    scope="Component/source and aggregation audit; NOT independent full framework end-to-end parity.",
    rgb_heldout_dataset_access="Geo2425/iMiGUE_SRTFD returns GatedRepoError 401; no access bypass attempted",
)
assert all(sources.values()) and diff < 1e-6 and diffhead < 1e-6
(O / "adapter_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
