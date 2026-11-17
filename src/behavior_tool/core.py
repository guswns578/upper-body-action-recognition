"""Video-independent bookkeeping and conservative geometric hand association."""

from collections import deque
import numpy as np
import cv2

HAND_LABELS = {
    "Closed_Fist": "주먹을 쥔 손 모양",
    "Open_Palm": "손바닥을 펼친 손 모양",
    "Pointing_Up": "검지를 위로 편 손 모양",
    "Thumb_Down": "엄지를 아래로 편 손 모양",
    "Thumb_Up": "엄지를 위로 편 손 모양",
    "Victory": "검지·중지를 편 V 손 모양",
    "ILoveYou": "엄지·검지·소지를 편 손 모양",
}


class CutDetector:
    """Heuristic hard-cut detector, not a learned shot detector; fades may be missed."""

    def __init__(self, threshold=0.18):
        self.previous = None
        self.hist = None
        self.threshold = threshold

    def update(self, frame):
        small = cv2.resize(frame, (64, 36))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([small], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3)
        cv2.normalize(hist, hist)
        mad = (
            0.0
            if self.previous is None
            else float(np.mean(np.abs(gray.astype(float) - self.previous)) / 255)
        )
        distance = (
            0.0
            if self.hist is None
            else float(cv2.compareHist(self.hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        )
        cut = self.previous is not None and (
            (mad > self.threshold and distance > 0.3) or mad > 0.38
        )
        self.previous = gray.astype(float)
        self.hist = hist
        return bool(cut), {"mean_pixel_change": mad, "histogram_distance": distance}


class Window:
    def __init__(self, seconds=2.5, gap=0.35):
        self.seconds = seconds
        self.gap = gap
        self.items = deque()
        self.last_emit = -1e9

    def clear(self):
        self.items.clear()
        self.last_emit = -1e9

    def add(self, t, points):
        if self.items and (t <= self.items[-1][0] or t - self.items[-1][0] > self.gap):
            self.clear()
        if points is None:
            self.clear()
            return
        self.items.append((t, points))
        while len(self.items) > 1 and self.items[1][0] <= t - self.seconds:
            self.items.popleft()

    def ready(self, t, interval=0.5):
        return (
            len(self.items) >= 12
            and self.items[-1][0] - self.items[0][0] >= self.seconds - 0.05
            and t - self.last_emit >= interval
        )


def upper_body_quality(xy, scores, width, height):
    # SimCC scores are recorded only: not visibility or calibrated confidence.
    pts = np.asarray(xy[:11])
    good = (
        np.isfinite(pts).all(1)
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )
    shoulder = float(np.linalg.norm(pts[5] - pts[6]))
    return bool(np.all(good[5:11]) and shoulder >= 12), good


class GeometryQuality:
    def __init__(self):
        self.previous = None

    def update(self, t, xy, scores, width, height, box):
        ok, good = upper_body_quality(xy, scores, width, height)
        p = np.asarray(xy)
        scale = max(1.0, float(np.linalg.norm(p[5] - p[6])))
        lengths = np.array(
            [np.linalg.norm(p[a] - p[b]) / scale for a, b in [(5, 7), (7, 9), (6, 8), (8, 10)]]
        )
        flags = []
        if not ok:
            flags.append("invalid_coordinates_or_small_shoulder_span")
        if np.any((lengths < 0.05) | (lengths > 3.0)):
            flags.append("implausible_projected_bone_length")
            ok = False
        x1, y1, x2, y2 = box
        for k in [7, 8, 9, 10]:
            if min(p[k, 0] - x1, x2 - p[k, 0], p[k, 1] - y1, y2 - p[k, 1]) < scale * 0.025:
                flags.append("joint_near_bbox_boundary")
                ok = False
                break
        if self.previous is not None:
            pt, old, oldlengths = self.previous
            if 0 < t - pt <= 0.15:
                jump = float(np.max(np.linalg.norm(p[5:11] - old[5:11], axis=1)) / scale)
                change = float(np.max(np.abs(lengths - oldlengths)))
                if jump > 0.8:
                    flags.append("large_coordinate_jump")
                if change > 0.6:
                    flags.append("projected_bone_length_change")
                # Fast coherent motion alone must not be discarded.
                if jump > 0.8 and change > 0.6:
                    ok = False
        self.previous = (t, p.copy(), lengths)
        return ok, flags


class ArmsCrossedRules:
    """Replaceable, unvalidated 2D posture candidate; never emotion inference."""

    def __init__(self):
        self.start = None
        self.last = None
        self.active = False

    def finish(self, t, reason):
        result = []
        if self.active:
            result = [
                {
                    "kind": "segment",
                    "label": "arms_crossed_candidate",
                    "description_ko": "팔짱 후보",
                    "start": self.start,
                    "end": self.last,
                    "ended_reason": reason,
                    "status": "unverified_candidate",
                }
            ]
        self.start = None
        self.last = None
        self.active = False
        return result

    def update(self, t, xy):
        if xy is None:
            return (
                self.finish(t, "unobservable")
                if self.last is not None and t - self.last > 0.15
                else []
            )
        p = np.asarray(xy)
        s = max(1.0, np.linalg.norm(p[5] - p[6]))
        axis = (p[6] - p[5]) / s
        mid = (p[5] + p[6]) / 2
        crossed = np.dot(p[9] - mid, axis) > 0.05 * s and np.dot(p[10] - mid, axis) < -0.05 * s
        compact = max(np.linalg.norm(p[9] - p[8]), np.linalg.norm(p[10] - p[7])) < 0.9 * s
        if crossed and compact:
            if self.start is None:
                self.start = t
            self.last = t
            self.active = t - self.start >= 0.3
            return []
        return self.finish(t, "posture_changed")


def assign_hands(xy, hand_points, box):
    """Associate crop-detected hands by wrists, ignoring mirror-sensitive handedness.

    Each hand must be clearly closer to one wrist, inside box, with no competing
    hand for that side. Returns indices, not proof of identity/visibility.
    """
    xy = np.asarray(xy)
    x1, y1, x2, y2 = box
    limit = max(15.0, min((y2 - y1) * 0.18, np.linalg.norm(xy[5] - xy[6]) * 0.65))
    candidates = {"left": [], "right": []}
    for i, points in enumerate(hand_points):
        wrist = np.asarray(points[0])
        if not (x1 <= wrist[0] <= x2 and y1 <= wrist[1] <= y2):
            continue
        distances = [float(np.linalg.norm(wrist - xy[9])), float(np.linalg.norm(wrist - xy[10]))]
        best = int(np.argmin(distances))
        other = 1 - best
        if distances[best] <= limit and distances[other] - distances[best] >= limit * 0.25:
            candidates[["left", "right"][best]].append(i)
    return {side: items[0] for side, items in candidates.items() if len(items) == 1}


def make_mmn_points(xy, assigned):
    """Require both independently detected hands; never hallucinate missing tips."""
    if set(assigned) != {"left", "right"}:
        return None
    p = np.asarray(xy)
    torso = np.stack(
        [p[0], (p[5] + p[6]) / 2, p[6], p[8], p[10], p[5], p[7], p[9], p[2], p[1], p[4], p[3]]
    )
    points = np.concatenate(
        [
            torso,
            np.asarray(assigned["right"])[[4, 8, 12, 16, 20]],
            np.asarray(assigned["left"])[[4, 8, 12, 16, 20]],
        ]
    )
    return points.astype(np.float32) if np.isfinite(points).all() else None
