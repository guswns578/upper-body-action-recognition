"""(gpu_env) MMA-52 영상을 겉모습(옷 색)으로 사람 단위로 묶는다. 공식 사람 ID가 없어서 개발셋을 사람 단위로 나누기 위함.

특징: 사람 박스를 위·가운데·아래 3등분해 HSV 색 히스토그램(8×4×4) → 제곱근 정규화, 영상 안 3프레임 평균.
군집: 코사인 거리 평균 연결(average linkage) 계층 군집, 지정한 개수로 자른다.
검증: 공식 val은 37명(논문 Table 1)이라 val에서 37개 군집이 깨끗한지 먼저 본다.

    python group_subjects.py val 37
    python group_subjects.py train 140
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
from scipy.cluster.hierarchy import linkage, fcluster

VID = Path('data/MMA-52/videos'); POSE = Path('data/MMA-52/pose_rtmw'); OUT = Path('data/MMA-52/subject_groups')


def feature(split, v):
    z = np.load(POSE/split/f'{v}.npz'); T = len(z['box']); c = cv2.VideoCapture(str(VID/split/f'{v}.mp4')); fs = []
    for t in [T//4, T//2, 3*T//4]:
        c.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = c.read()
        if not ok:
            continue
        x1, y1, x2, y2 = z['box'][t].astype(int); crop = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2HSV); h = crop.shape[0]
        parts = [crop[int(h*a):int(h*b)] for a, b in [(0, .33), (.33, .62), (.62, 1)]]
        hs = [cv2.calcHist([p], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel() for p in parts]
        fs.append(np.concatenate([np.sqrt(x/max(x.sum(), 1)) for x in hs]))
    c.release()
    return np.mean(fs, 0)


def main():
    split, k = sys.argv[1], int(sys.argv[2]); OUT.mkdir(exist_ok=True)
    vids = sorted(p.stem for p in (POSE/split).glob('*.npz'))
    X = np.stack([feature(split, v) for v in vids]); X /= np.linalg.norm(X, axis=1, keepdims=True)
    lab = fcluster(linkage(X, 'average', metric='cosine'), k, 'maxclust')
    sizes = np.bincount(lab)[1:]
    print(f'{split}: {len(vids)} videos → {k} groups, size min {sizes.min()} median {int(np.median(sizes))} max {sizes.max()}')
    (OUT/f'{split}_groups.json').write_text(json.dumps(dict(zip(vids, lab.tolist())), indent=0), encoding='utf-8')
    np.save(OUT/f'{split}_features.npy', X)
    # 검증용: 군집마다 영상 6개 중간 프레임을 한 줄로
    rows = []
    for g in np.argsort(-sizes)[:20]+1:
        members = [v for v, l in zip(vids, lab) if l == g][:8]; tiles = []
        for v in members:
            z = np.load(POSE/split/f'{v}.npz'); T = len(z['box']); c = cv2.VideoCapture(str(VID/split/f'{v}.mp4'))
            c.set(cv2.CAP_PROP_POS_FRAMES, T//2); ok, f = c.read(); c.release(); x1, y1, x2, y2 = z['box'][T//2].astype(int)
            tiles.append(cv2.resize(f[y1:y2, x1:x2], (60, 100)))
        tiles += [np.zeros((100, 60, 3), np.uint8)]*(8-len(tiles))
        row = np.hstack(tiles); cv2.putText(row, f'g{g} n={sizes[g-1]}', (2, 12), 0, .4, (0, 0, 255), 1); rows.append(row)
    cv2.imwrite(str(OUT/f'{split}_groups_preview.jpg'), np.vstack([np.hstack(rows[i:i+2]) for i in range(0, len(rows), 2)]))


if __name__ == '__main__':
    main()
