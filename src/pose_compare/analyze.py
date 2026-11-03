"""(pose_env) 정답 없이 계산 가능한 포즈 추출기 비교 지표 + 시각 비교 이미지.

지표(모두 어깨 너비 S로 정규화, S = 프레임별 '참조' 양어깨 거리의 중앙값):
- missing: 관절 미출력 비율(NaN). RTMPose/RTMW/MediaPipe는 항상 출력하므로 0에 가깝다(점수는 별도).
- jitter: |p[t+1] - 2p[t] + p[t-1]| / S 의 중앙값·95백분위. 같은 영상이므로 실제 움직임은 모델 간 동일하고,
  차이는 추정 잡음을 반영한다(값이 작을수록 안정). 단 평활이 들어간 모델(MediaPipe VIDEO)은 지연과 맞바꾼 값이다.
- dev: 자기 자신을 뺀 다른 계열 모델들의 관절별 중앙값(참조)과의 거리 / S. 정답 오차가 아니라 '합의에서 벗어난 정도'.
- swap: 참조 대비 좌우 손목(또는 팔꿈치)이 뒤바뀐 프레임 비율.
- ms: 프레임당 포즈 추론 시간 중앙값(사람 검출 제외).
"""
import json
import numpy as np
import cv2
from common import HERE, CLIPS, RAW, UPPER, GROUPS, SOURCES, read_frames

MODELS = ['openpose', 'rtmpose_m', 'rtmw_l', 'mediapipe_image', 'mediapipe_video']
FAMILY = {'openpose': 'op', 'rtmpose_m': 'rtm', 'rtmw_l': 'rtmw', 'mediapipe_image': 'mp', 'mediapipe_video': 'mp'}
REF_POOL = ['openpose', 'rtmpose_m', 'rtmw_l', 'mediapipe_image']   # 계열별 1개씩 참조 후보
LABEL = {'openpose': 'OpenPose (COCO-18)', 'rtmpose_m': 'RTMPose-m (17)', 'rtmw_l': 'RTMW-l (133)',
         'mediapipe_image': 'MediaPipe heavy IMAGE', 'mediapipe_video': 'MediaPipe heavy VIDEO'}
COLOR = {'openpose': (0, 140, 255), 'rtmpose_m': (60, 200, 60), 'rtmw_l': (220, 60, 220), 'mediapipe_image': (255, 120, 0)}
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4)]


def load(clip):
    d = {}
    for m in MODELS:
        f = RAW/f'{clip}__{m}.npz'
        if f.exists():
            z = np.load(f); d[m] = dict(xy=z['xy'].astype(np.float64), score=z['score'], ms=z['ms'])
    return d


def reference(d, exclude):
    pool = [d[m]['xy'] for m in REF_POOL if m in d and FAMILY[m] != FAMILY[exclude]]
    return np.nanmedian(np.stack(pool), axis=0) if pool else None


def metrics(d):
    allref = np.nanmedian(np.stack([d[m]['xy'] for m in REF_POOL if m in d]), axis=0)
    S = np.nanmedian(np.linalg.norm(allref[:, 5]-allref[:, 6], axis=-1))
    out = {}
    for m, v in d.items():
        xy = v['xy']; ref = reference(d, m); r = {'frames': int(len(xy)), 'shoulder_px': float(S)}
        miss = np.isnan(xy[..., 0])
        acc = np.linalg.norm(xy[2:]-2*xy[1:-1]+xy[:-2], axis=-1)/S
        dev = np.linalg.norm(xy-ref, axis=-1)/S if ref is not None else None
        for g, idx in GROUPS.items():
            r[f'missing_{g}'] = float(miss[:, idx].mean())
            a = acc[:, idx]; a = a[np.isfinite(a)]
            r[f'jitter_{g}_med'] = float(np.median(a)) if a.size else None
            r[f'jitter_{g}_p95'] = float(np.percentile(a, 95)) if a.size else None
            if dev is not None:
                e = dev[:, idx]; e = e[np.isfinite(e)]
                r[f'dev_{g}_med'] = float(np.median(e)) if e.size else None
                r[f'dev_{g}_p95'] = float(np.percentile(e, 95)) if e.size else None
        if ref is not None:
            for name, (li, ri) in {'wrist': (9, 10), 'elbow': (7, 8)}.items():
                same = np.linalg.norm(xy[:, li]-ref[:, li], axis=-1)+np.linalg.norm(xy[:, ri]-ref[:, ri], axis=-1)
                cross = np.linalg.norm(xy[:, li]-ref[:, ri], axis=-1)+np.linalg.norm(xy[:, ri]-ref[:, li], axis=-1)
                ok = np.isfinite(same) & np.isfinite(cross) & (np.linalg.norm(ref[:, li]-ref[:, ri], axis=-1) > 0.3*S)
                r[f'swap_{name}'] = float((cross[ok] < 0.7*same[ok]).mean()) if ok.any() else None
        ms = v['ms'][v['ms'] > 0]; r['ms_med'] = float(np.median(ms)) if ms.size else None
        out[m] = r
    return out


def contact_sheet(clip, d, n=4):
    c = np.load(CLIPS/f'{clip}.npz'); T = len(c['boxes']); fps = float(c['fps'])
    picks = np.linspace(T*0.1, T*0.9, n).astype(int)
    rows = []
    frames = {t: f for t, _, f in read_frames(c['path'].item(), float(c['start_s']), float(c['start_s'])+T/fps+1e-6) if t in set(picks)}
    for t in picks:
        b = c['boxes'][t]; W, H = c['size']
        x1, y1 = int(max(0, b[0]-0.15*(b[2]-b[0]))), int(max(0, b[1]-0.1*(b[3]-b[1])))
        x2, y2 = int(min(W, b[2]+0.15*(b[2]-b[0]))), int(min(H, b[3]+0.1*(b[3]-b[1])))
        tiles = []
        for m in REF_POOL:
            img = frames[t][y1:y2, x1:x2].copy(); sc = 360/img.shape[0]; img = cv2.resize(img, None, fx=sc, fy=sc)
            if m in d:
                p = (d[m]['xy'][t]-[x1, y1])*sc
                for a_, b_ in EDGES:
                    if np.isfinite(p[[a_, b_]]).all():
                        cv2.line(img, tuple(p[a_].astype(int)), tuple(p[b_].astype(int)), COLOR[m], 3, cv2.LINE_AA)
                for k in range(13):
                    if np.isfinite(p[k]).all():
                        cv2.circle(img, tuple(p[k].astype(int)), 5 if k in (9, 10) else 3, (255, 255, 255), -1)
                # 오른손목은 R 표시(해부학적 오른쪽)
                if np.isfinite(p[10]).all():
                    cv2.putText(img, 'R', tuple((p[10]+[6, -6]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            img = letterbox(img, 300, 360)
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, f'{m}  t={t/fps:.1f}s', (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        rows.append(np.hstack(tiles))
    (HERE/'figs').mkdir(exist_ok=True)
    cv2.imwrite(str(HERE/'figs'/f'{clip}.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def letterbox(img, w, h):
    """비율을 유지해 w×h 칸에 맞추고 남는 곳은 검게 채운다(타일 크기 통일)."""
    s = min(w/img.shape[1], h/img.shape[0]); img = cv2.resize(img, None, fx=s, fy=s)
    out = np.zeros((h, w, 3), np.uint8); y, x = (h-img.shape[0])//2, (w-img.shape[1])//2
    out[y:y+img.shape[0], x:x+img.shape[1]] = img
    return out


def main():
    report = {}
    for name, *_ in SOURCES:
        d = load(name)
        if len(d) < 2:
            continue
        report[name] = metrics(d); contact_sheet(name, d)
    # 프레임 가중 평균
    agg = {}
    for m in MODELS:
        rows = [r[m] for r in report.values() if m in r]
        if not rows:
            continue
        keys = [k for k in rows[0] if k not in ('frames', 'shoulder_px')]
        agg[m] = {k: float(np.average([r[k] for r in rows if r.get(k) is not None],
                                       weights=[r['frames'] for r in rows if r.get(k) is not None]))
                  for k in keys if any(r.get(k) is not None for r in rows)}
        agg[m]['clips'] = len(rows); agg[m]['frames'] = int(sum(r['frames'] for r in rows))
    (HERE/'metrics.json').write_text(json.dumps({'per_clip': report, 'aggregate': agg}, indent=1), encoding='utf-8')
    cols = ['clips', 'frames', 'missing_wrist', 'jitter_wrist_med', 'jitter_elbow_med', 'jitter_shoulder_med',
            'dev_wrist_med', 'dev_wrist_p95', 'dev_elbow_med', 'swap_wrist', 'ms_med']
    print('model'.ljust(18)+''.join(c[:16].rjust(17) for c in cols))
    for m, r in agg.items():
        print(m.ljust(18)+''.join((f'{r[c]:.3f}' if isinstance(r.get(c), float) else str(r.get(c))).rjust(17) for c in cols))


if __name__ == '__main__':
    main()
