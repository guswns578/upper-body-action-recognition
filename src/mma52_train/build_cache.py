"""(gpu_env) MMA-52 RTMW 좌표 → 학습 캐시. 상반신 13점 + 손 42점, 프레임별 40종 라벨.

정규화(모두 인과적 — 현재 프레임까지만 사용):
- 원점: 프레임별 양어깨 중점.
- 크기: 현재 프레임까지 최근 60프레임(2초) 어깨 너비의 중앙값. 영상 시작부는 있는 만큼만 쓴다.
- 손: 같은 원점·크기로 정규화(손목 상대가 아니라 몸 기준; 손목 상대 좌표는 학습 때 계산 가능).
라벨: 다리·발 12종 제외 40종. y[t,c]=1 ⇔ t ∈ [start_frame, end_frame] (끝 포함). 끝은 실제 프레임 수로 자른다.
"""
import csv, json
from pathlib import Path
import numpy as np

POSE = Path('data/MMA-52/pose_rtmw'); ANN = Path('data/MMA-52/annotations'); OUT = Path('data/MMA-52/cache_v1')
LEG = {24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51}
KEEP = [c for c in range(52) if c not in LEG]                       # 40종, 원래 번호
UPPER = list(range(13))                                             # COCO 0~12
HANDS = list(range(91, 133))
WIN = 60


def causal_rolling_median(x, w):
    out = np.empty_like(x)
    for t in range(len(x)):
        out[t] = np.median(x[max(0, t-w+1):t+1])
    return out


def main():
    OUT.mkdir(exist_ok=True)
    names = [l.split('\t')[1].strip() for l in open(ANN/'label_name.txt', encoding='utf-8') if l.strip()]
    for sp in ['train', 'val']:
        rows = list(csv.DictReader(open(ANN/f'{sp}.csv')))
        segs = {}
        for r in rows:
            segs.setdefault(r['video_id'], []).append((int(r['start_frame']), int(r['end_frame']), int(r['class'])))
        vids = sorted(segs)
        body, score, hands, hscore, lab, off, fps = [], [], [], [], [], [0], []
        for v in vids:
            z = np.load(POSE/sp/f'{v}.npz'); xy = z['xy'].astype(np.float64); sc = z['score'].astype(np.float32); T = len(xy)
            mid = (xy[:, 5]+xy[:, 6])/2
            sw = np.linalg.norm(xy[:, 5]-xy[:, 6], axis=-1)
            scale = np.maximum(causal_rolling_median(sw, WIN), 1.0)[:, None, None]
            body.append(((xy[:, UPPER]-mid[:, None])/scale).astype(np.float32)); score.append(sc[:, UPPER])
            hands.append(((xy[:, HANDS]-mid[:, None])/scale).astype(np.float16)); hscore.append(sc[:, HANDS].astype(np.float16))
            y = np.zeros((T, 52), bool)
            for s, e, c in segs[v]:
                y[s:min(e, T-1)+1, c] = True
            lab.append(y[:, KEEP]); off.append(off[-1]+T); fps.append(float(z['fps']))
        np.savez(OUT/f'{sp}.npz', body=np.concatenate(body), score=np.concatenate(score), hands=np.concatenate(hands),
                 hscore=np.concatenate(hscore), labels=np.concatenate(lab), offsets=np.array(off), fps=np.array(fps),
                 videos=np.array(vids))
        L = np.concatenate(lab)
        print(f'{sp}: {len(vids)} videos, {len(L)} frames, frames with any of 40: {L.any(1).mean():.3f}', flush=True)
    (OUT/'meta.json').write_text(json.dumps({'keep_original_ids': KEEP, 'names': [names[c] for c in KEEP], 'upper': UPPER,
        'hands': HANDS, 'scale_window_frames': WIN, 'label_rule': 'inclusive end, clipped to decoded frames'}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
