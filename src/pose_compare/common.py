"""포즈 추출기 비교 공통 정의 (환경 독립: numpy, cv2만 사용).

모든 추출기는 같은 영상 구간, 같은 사람 박스(YOLOX tiny)를 받는다.
출력은 공통 상반신 13점(해부학적 좌우, 원본 영상 픽셀 좌표)으로 바꿔 저장한다.
"""

from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
RAW = HERE / "raw"

UPPER = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
]
# 각 모델의 관절 인덱스 → UPPER 순서
MAP = {
    "coco17": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # RTMPose, RTMW 앞 17점
    "openpose18": [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8],  # COCO-18 (1=neck 제외)
    "mediapipe33": [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24],
}
GROUPS = {
    "head": [0, 1, 2, 3, 4],
    "shoulder": [5, 6],
    "elbow": [7, 8],
    "wrist": [9, 10],
    "hip": [11, 12],
}

# 영상 구간: (이름, 경로, 시작초, 끝초). 끝초 None이면 끝까지.
SOURCES = [
    ("px5940726", ROOT / "src/seated_references/5940726.mp4", 0, None),
    ("px8189172", ROOT / "src/seated_references/8189172.mp4", 0, None),
    ("px6602739", ROOT / "src/seated_references/6602739.mp4", 0, None),
    ("px6100890", ROOT / "src/seated_references/6100890.mp4", 0, 12),
    ("px7671778", ROOT / "src/arms_generalization/7671778.mp4", 0, None),
    ("px8419422", ROOT / "src/arms_generalization/8419422.mp4", 0, None),
    ("emph44_70_92", Path("data/aihub/emphasis_gesture_44.mp4"), 70, 92),
]


def read_frames(path, start_s, end_s):
    """BGR 프레임 생성기. (로컬 프레임 번호, 원본 프레임 번호, 프레임)"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(round(start_s * fps))
    last = int(round(end_s * fps)) if end_s is not None else 10**9
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    i = first
    while i < last:
        ok, f = cap.read()
        if not ok:
            break
        yield i - first, i, f
        i += 1
    cap.release()


def expand(box, w, h, ratio=1.25):
    """박스를 중심 기준 ratio배 확장 후 영상 안으로 자른다(크롭 입력 모델용)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * ratio, (y2 - y1) * ratio
    return np.array(
        [max(0, cx - bw / 2), max(0, cy - bh / 2), min(w - 1, cx + bw / 2), min(h - 1, cy + bh / 2)]
    )


def save_raw(model, clip, xy, score, ms, extra=None):
    """xy [T,13,2] 원본 좌표(결측 NaN), score [T,13] 모델 고유 점수, ms [T] 프레임당 추론 시간."""
    RAW.mkdir(exist_ok=True)
    np.savez(
        RAW / f"{clip}__{model}.npz",
        xy=xy.astype(np.float32),
        score=score.astype(np.float32),
        ms=np.asarray(ms, np.float32),
        **(extra or {}),
    )
