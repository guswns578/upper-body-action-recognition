"""(gpu_env) 머리 동작용 얼굴 특징 캐시. cache_v1과 같은 영상 순서·프레임 수로 저장한다.

- 얼굴 68점(COCO-WholeBody 23~90) 중 17점: 턱 0,4,8,12,16 / 눈썹 19,24 / 코 27,30,33 / 눈꼬리 36,39,42,45 / 입 48,54,57.
- face_local: 얼굴 중심(17점 평균) 기준, 양 눈 바깥 꼬리(36–45) 거리의 과거 2초 중앙값으로 나눈 좌표.
  머리가 작게 찍혀도 고개 숙임·돌림·기울임이 얼굴 안 점들의 배치 변화로 크게 드러나게 하려는 것.
- face_center: 얼굴 중심을 양어깨 중점 기준, 어깨 너비(cache_v1과 같은 과거 2초 중앙값)로 나눈 좌표.
- face_score: 17점 점수.
"""
from pathlib import Path
import numpy as np
from build_cache import causal_rolling_median, WIN

POSE = Path('data/MMA-52/pose_rtmw'); CACHE = Path('data/MMA-52/cache_v1')
SEL = [0, 4, 8, 12, 16, 19, 24, 27, 30, 33, 36, 39, 42, 45, 48, 54, 57]
IDX = [23+i for i in SEL]


def main():
    for sp in ['train', 'val']:
        c = np.load(CACHE/f'{sp}.npz'); vids, off = c['videos'], c['offsets']
        loc, cen, fsc = [], [], []
        for k, v in enumerate(vids):
            z = np.load(POSE/sp/f'{v}.npz'); xy = z['xy'].astype(np.float64); sc = z['score'].astype(np.float32)
            assert len(xy) == off[k+1]-off[k]
            f = xy[:, IDX]; fc = f.mean(1)
            eye = np.maximum(causal_rolling_median(np.linalg.norm(xy[:, 23+36]-xy[:, 23+45], axis=-1), WIN), 1.0)
            mid = (xy[:, 5]+xy[:, 6])/2
            sw = np.maximum(causal_rolling_median(np.linalg.norm(xy[:, 5]-xy[:, 6], axis=-1), WIN), 1.0)
            loc.append(((f-fc[:, None])/eye[:, None, None]).astype(np.float32))
            cen.append(((fc-mid)/sw[:, None]).astype(np.float32)); fsc.append(sc[:, IDX])
        np.savez(CACHE/f'{sp}_face.npz', face_local=np.concatenate(loc), face_center=np.concatenate(cen), face_score=np.concatenate(fsc))
        L = np.concatenate(loc); print(f'{sp}: {len(L)} frames, face_local |x| p99 {np.percentile(np.abs(L), 99):.2f}', flush=True)


if __name__ == '__main__':
    main()
