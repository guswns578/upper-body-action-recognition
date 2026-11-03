"""(gpu_env) MMA-52 상반신 13점 → 프레임별 40종 다중 라벨. 과거만 보는(인과적) 스트리밍 모델.

    python train.py --model tcn --seed 0            # 기준 모델
    python train.py --model gru --seed 0
    python train.py --check                          # E0: 인과성·한 배치 과적합 점검

모델 선택은 train에서 떼어 낸 개발셋(영상 10%)으로 하고, 공식 val은 마지막에 한 번 평가한다.
주의: MMA-52 공식 train/val은 사람 단위로 나뉘지만 사람 ID가 없어 개발셋은 train과 같은 사람을 포함할 수 있다.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

CACHE = Path('data/MMA-52/cache_v1'); HERE = Path(__file__).resolve().parent
V0 = Path('data/label_schema_v0/labels_v0.json')
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]          # 13점 중 좌우 쌍
HIPS = [11, 12]
HAND_PAIR = (13, 14)                                                 # --hands: 손 21점 평균(왼손 13, 오른손 14)
HEAD_CLASSES = ['bowing head', 'head up', 'tilting head', 'turning head', 'nodding', 'shaking head']
FACE_PAIRS = [(0, 4), (1, 3), (5, 6), (10, 13), (11, 12), (14, 15)]   # build_face_cache.SEL 17점 중 좌우 쌍


# ---------------------------------------------------------------- 데이터
class Split:
    def __init__(self, name, face=False, hands=False):
        z = np.load(CACHE/f'{name}.npz')
        self.body, self.score, self.labels = z['body'], z['score'], z['labels']
        if hands:   # 손가락 모양이 아니라 '손이 어디 있나': 손 21점 평균 위치·평균 점수를 관절 2개로 덧붙인다(15점)
            h, hs = z['hands'].astype(np.float32).reshape(-1, 2, 21, 2), z['hscore'].astype(np.float32).reshape(-1, 2, 21)
            self.body = np.concatenate([self.body, h.mean(2)], 1); self.score = np.concatenate([self.score, hs.mean(2)], 1)
        self.off, self.fps, self.videos = z['offsets'], z['fps'], z['videos']
        self.face = None
        if face:
            f = np.load(CACHE/f'{name}_face.npz'); self.face = (f['face_local'], f['face_center'], f['face_score'])

    def __len__(self):
        return len(self.videos)

    def get(self, i):
        s, e = self.off[i], self.off[i+1]
        return self.body[s:e], self.score[s:e], self.labels[s:e]

    def get_face(self, i):
        s, e = self.off[i], self.off[i+1]
        return [a[s:e] for a in self.face]


def batch_crops(split, idx, L, rng):
    B, J = len(idx), split.body.shape[1]; xy = np.zeros((B, L, J, 2), np.float32); sc = np.zeros((B, L, J), np.float32)
    y = np.zeros((B, L, 40), np.float32); m = np.zeros((B, L), np.float32)
    fl, fc, fs = np.zeros((B, L, 17, 2), np.float32), np.zeros((B, L, 2), np.float32), np.zeros((B, L, 17), np.float32)
    for b, i in enumerate(idx):
        bx, bs, by = split.get(i); T = len(bx); s = rng.integers(0, max(1, T-L+1)); n = min(L, T-s)
        xy[b, :n], sc[b, :n], y[b, :n], m[b, :n] = bx[s:s+n], bs[s:s+n], by[s:s+n], 1
        if split.face is not None:
            a, c, d = split.get_face(i); fl[b, :n], fc[b, :n], fs[b, :n] = a[s:s+n], c[s:s+n], d[s:s+n]
    out = [torch.from_numpy(a) for a in (xy, sc, y, m)]
    face = [torch.from_numpy(a) for a in (fl, fc, fs)] if split.face is not None else None
    return out, face


def augment(xy, sc, rng, face=None):
    """좌우 반전(관절 쌍 교환), 회전 ±10°, 크기 ±10%, 골반 가림(책상), 관절 무작위 결측, 좌표 잡음.
    face=(face_local, face_center, face_score)가 있으면 같은 반전·회전을 적용한다(크기는 얼굴 중심 위치에만)."""
    B = xy.shape[0]; dev = xy.device
    flip = torch.from_numpy(rng.random(B) < .5).to(dev)
    if flip.any():
        f = xy[flip].clone(); f[..., 0] *= -1; g = sc[flip].clone()
        for a, b in PAIRS+([HAND_PAIR] if xy.shape[2] > 13 else []):
            f[:, :, [a, b]] = f[:, :, [b, a]]; g[:, :, [a, b]] = g[:, :, [b, a]]
        xy[flip], sc[flip] = f, g
        if face is not None:
            fl, fc, fs = [t[flip].clone() for t in face]; fl[..., 0] *= -1; fc[..., 0] *= -1
            for a, b in FACE_PAIRS:
                fl[:, :, [a, b]] = fl[:, :, [b, a]]; fs[:, :, [a, b]] = fs[:, :, [b, a]]
            face[0][flip], face[1][flip], face[2][flip] = fl, fc, fs
    ang = torch.from_numpy(rng.uniform(-.17, .17, B)).float().to(dev)
    R = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], -2)
    scale = torch.from_numpy(rng.uniform(.9, 1.1, B)).float().to(dev)
    xy = torch.einsum('bij,btkj->btki', R, xy)*scale[:, None, None, None]
    if face is not None:
        face[0] = torch.einsum('bij,btkj->btki', R, face[0])+torch.randn_like(face[0])*.01
        face[1] = torch.einsum('bij,btj->bti', R, face[1])*scale[:, None, None]
    for b in np.nonzero(rng.random(B) < .3)[0]:     # 30% 표본은 책상에 가린 것처럼 골반을 결측 처리
        xy[b, :, HIPS] = 0; sc[b, :, HIPS] = 0
    drop = torch.rand(sc.shape, device=dev) < .03
    xy = xy.masked_fill(drop[..., None], 0); sc = sc.masked_fill(drop, 0)
    xy = xy+torch.randn_like(xy)*.01
    return xy, sc, face


def diff(a):
    return torch.cat([torch.zeros_like(a[:, :1]), a[:, 1:]-a[:, :-1]], 1)


def features(xy, sc, stats, face=None):
    """[B,T,13,2],[B,T,13] → [B,T,65]: 좌표 26 + 속도 26(과거와의 차) + 표준화 점수 13.
    face가 있으면 + 얼굴 17점 26·34 → 얼굴 국소 좌표 34 + 속도 34 + 얼굴 중심 2 + 속도 2 + 표준화 점수 17 = 89 → 총 154."""
    s = (sc-stats[0])/stats[1]
    B, T = xy.shape[:2]
    x = [xy.reshape(B, T, -1), diff(xy).reshape(B, T, -1)*10, s]
    if face is not None:
        fl, fc, fs = face
        x += [fl.reshape(B, T, -1), diff(fl).reshape(B, T, -1)*10, fc, diff(fc)*10, (fs-stats[0])/stats[1]]
    return torch.cat(x, -1)


# ---------------------------------------------------------------- 모델
class CausalTCN(nn.Module):
    def __init__(self, d_in=65, h=128, n_out=40, dil=(1, 2, 4, 8, 16, 32), drop=.1):
        super().__init__()
        self.inp = nn.Conv1d(d_in, h, 1)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(conv=nn.Conv1d(h, h, 3, dilation=d), norm=nn.BatchNorm1d(h),
                                                        proj=nn.Conv1d(h, h, 1))) for d in dil])
        self.dil, self.drop, self.out = dil, nn.Dropout(drop), nn.Conv1d(h, n_out, 1)
        self.receptive = 1+2*sum(dil)

    def forward(self, x):                       # x [B,T,D] → logits [B,T,C]
        z = self.inp(x.transpose(1, 2))
        for d, b in zip(self.dil, self.blocks):
            u = b['conv'](F.pad(z, (2*d, 0)))   # 왼쪽(과거)만 패딩 → 미래를 보지 않음
            z = z+b['proj'](self.drop(F.gelu(b['norm'](u))))
        return self.out(z).transpose(1, 2)


class TwoStreamTCN(nn.Module):
    """몸(앞 65차원)과 얼굴(뒤 89차원)을 각자의 인과적 TCN으로 처리한 뒤 프레임별로 이어 붙여 판정한다.
    한 덩어리로 넣었을 때 얼굴 입력이 팔 정보를 밀어내는지 확인하려는 구조."""
    def __init__(self, d_body=65, d_face=89, h=96, n_out=40, drop=.1):
        super().__init__()
        self.d_body = d_body
        self.body, self.face = CausalTCN(d_body, h, h, drop=drop), CausalTCN(d_face, h, h, drop=drop)
        self.out = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(2*h, n_out))
        self.receptive = self.body.receptive

    def forward(self, x):
        return self.out(torch.cat([self.body(x[..., :self.d_body]), self.face(x[..., self.d_body:])], -1))


class CausalGRU(nn.Module):
    def __init__(self, d_in=65, h=192, n_out=40, layers=2, drop=.1):
        super().__init__()
        self.inp = nn.Linear(d_in, h); self.gru = nn.GRU(h, h, layers, batch_first=True, dropout=drop)
        self.out = nn.Linear(h, n_out); self.receptive = None

    def forward(self, x):
        return self.out(self.gru(F.gelu(self.inp(x)))[0])


# ---------------------------------------------------------------- 평가
def average_precision(y, p):
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind='stable'); y = y[o]; tp = np.cumsum(y)
    return float((tp[y > 0]/(np.nonzero(y)[0]+1)).mean())


@torch.no_grad()
def predict(model, split, stats, dev, idx=None):
    model.eval(); idx = range(len(split)) if idx is None else idx; P, Y, F_ = [], [], []
    for i in idx:
        bx, bs, by = split.get(i)
        face = [torch.from_numpy(a)[None].to(dev) for a in split.get_face(i)] if split.face is not None else None
        x = features(torch.from_numpy(bx)[None].to(dev), torch.from_numpy(bs)[None].to(dev), stats, face)
        P.append(torch.sigmoid(model(x))[0].float().cpu().numpy()); Y.append(by); F_.append(np.full(len(by), split.fps[i]))
    return P, Y, F_


def segments(y):
    d = np.diff(np.concatenate([[0], y.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]-1))


def evaluate(P, Y, Fps, names, eval_classes, thresholds=None):
    p, y = np.concatenate(P), np.concatenate(Y).astype(np.float32)
    ap = {names[c]: average_precision(y[:, c], p[:, c]) for c in range(40)}
    res = {'mAP_eval_classes': float(np.nanmean([ap[names[c]] for c in eval_classes])),
           'mAP_all_with_positives': float(np.nanmean(list(ap.values()))), 'AP': ap}
    # 라벨 v0로 묶은 AP: 정답 = 대응 클래스 중 하나라도 1, 점수 = 대응 클래스 최댓값
    v0 = json.loads(V0.read_text(encoding='utf-8')); keep = json.loads((CACHE/'meta.json').read_text())['keep_original_ids']
    grouped = {}
    for code, m in v0['public_dataset_mapping'].items():
        if code == 'note':
            continue
        ids = [keep.index(i) for i, _, _ in m.get('mma52', []) if i in keep]
        if ids:
            grouped[code] = average_precision(y[:, ids].max(1), p[:, ids].max(1))
    res['AP_v0_grouped'] = grouped
    # 상반신 라벨이 전혀 없는 프레임의 오검출 비율(확률 0.5 기준, 참고용: '없음'이 확실한 음성인지 보장 안 됨)
    none = y.sum(1) == 0
    res['none_frames_fraction'] = float(none.mean()); res['false_alarm_rate_at_0.5_on_none_frames'] = float((p[none].max(1) > .5).mean())
    # 시작 지연: 클래스별 임계값(개발셋 F1 최대)을 넘는 첫 프레임까지 걸린 시간
    if thresholds is not None:
        lat = {}
        for c in eval_classes:
            ds, miss = [], 0
            for pp, yy, ff in zip(P, Y, Fps):
                for s, e in segments(yy[:, c]):
                    hit = np.nonzero(pp[s:e+1, c] > thresholds[c])[0]
                    if hit.size:
                        ds.append(hit[0]/ff[0])
                    else:
                        miss += 1
            lat[names[c]] = {'median_s': float(np.median(ds)) if ds else None, 'detected': len(ds), 'missed': miss}
        res['onset_latency'] = lat
    return res


def best_thresholds(P, Y):
    p, y = np.concatenate(P), np.concatenate(Y)
    th = []
    for c in range(40):
        best, bt = -1, .5
        for t in np.linspace(.05, .95, 19):
            q = p[:, c] > t; tp = (q & y[:, c]).sum(); f1 = 2*tp/max(1, q.sum()+y[:, c].sum())
            if f1 > best:
                best, bt = f1, t
        th.append(float(bt))
    return th


# ---------------------------------------------------------------- 실행
def build(name):
    return CausalTCN() if name == 'tcn' else CausalGRU()


def check(dev):
    """인과성: 미래 프레임을 바꿔도 과거 출력이 그대로인지. 과적합: 한 배치를 외울 수 있는지."""
    tr = Split('train'); rng = np.random.default_rng(0)
    stats = torch.tensor([tr.score.mean(), tr.score.std()], device=dev)
    for name in ['tcn', 'gru']:
        m = build(name).to(dev).eval()
        x = torch.randn(2, 200, 65, device=dev); x2 = x.clone(); x2[:, 120:] = torch.randn_like(x2[:, 120:])
        with torch.no_grad():
            d = (m(x)[:, :120]-m(x2)[:, :120]).abs().max().item(); d2 = (m(x)[:, 120:]-m(x2)[:, 120:]).abs().max().item()
        print(f'[{name}] causality: past-output change {d:.2e} (must be 0), future-output change {d2:.2e} (should be >0)')
        m.train(); opt = torch.optim.AdamW(m.parameters(), 3e-3)
        xy, sc, y, mk = [a.to(dev) for a in batch_crops(tr, rng.choice(len(tr), 8, replace=False), 128, rng)[0]]
        for it in range(300):
            lo = (F.binary_cross_entropy_with_logits(m(features(xy, sc, stats)), y, reduction='none').mean(-1)*mk).sum()/mk.sum()
            opt.zero_grad(); lo.backward(); opt.step()
        with torch.no_grad():
            acc = ((m(features(xy, sc, stats)) > 0).float() == y).float()[mk > 0].mean().item()
        print(f'[{name}] overfit one batch: loss {lo.item():.4f}, frame-label accuracy {acc:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tcn'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--check', action='store_true'); ap.add_argument('--tag', default='')
    ap.add_argument('--dev-split', default='group', choices=['group', 'random'])
    ap.add_argument('--hidden', type=int, default=128); ap.add_argument('--dropout', type=float, default=.1)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--face', action='store_true', help='얼굴 17점 특징 추가(build_face_cache.py)')
    ap.add_argument('--hands', action='store_true', help='손 중심 2점 추가(손가락 모양은 쓰지 않음)')
    ap.add_argument('--no-head', action='store_true', help='머리 동작 6종을 학습 손실·평가에서 뺀다')
    a = ap.parse_args(); dev = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.check:
        return check(dev)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr, va = Split('train', a.face, a.hands), Split('val', a.face, a.hands); names = json.loads((CACHE/'meta.json').read_text())['names']
    if a.dev_split == 'group':   # 겉모습 군집(group_subjects.py) 단위로 떼어 내 같은 사람이 train·dev에 걸치지 않게 함
        grp = json.loads(Path('data/MMA-52/subject_groups/train_groups.json').read_text()); g = np.array([grp[v] for v in tr.videos])
        order, held = np.random.default_rng(12345).permutation(np.unique(g)), []
        for k in order:
            held.append(k)
            if np.isin(g, held).sum() >= len(tr)*.12:
                break
        dev_mask = np.isin(g, held)
    else:
        dev_mask = np.zeros(len(tr), bool); dev_mask[np.random.default_rng(12345).permutation(len(tr))[:len(tr)//10]] = True
    dev_idx, tr_idx = np.nonzero(dev_mask)[0], np.nonzero(~dev_mask)[0]
    print(f'dev split={a.dev_split}: train {len(tr_idx)} / dev {len(dev_idx)} videos', flush=True)
    stats = torch.tensor([tr.score.mean(), tr.score.std()], device=dev)
    # 평가 대상: 공식 val에서 구간 10개 이상인 클래스
    seg_counts = [sum(len(segments(va.get(i)[2][:, c])) for i in range(len(va))) for c in range(40)]
    eval_classes = [c for c in range(40) if seg_counts[c] >= 10 and not (a.no_head and names[c] in HEAD_CLASSES)]
    cls_w = torch.tensor([0. if a.no_head and n in HEAD_CLASSES else 1. for n in names], device=dev)
    y_tr = np.concatenate([tr.get(i)[2] for i in tr_idx]); pos = y_tr.mean(0)
    pos_w = torch.tensor(np.clip((1-pos)/np.maximum(pos, 1e-6), 1, 20), dtype=torch.float32, device=dev)
    d_in = 5*tr.body.shape[1]+(89 if a.face else 0)
    if a.model == 'tcn2':
        assert a.face, 'tcn2는 --face가 필요하다'
        model = TwoStreamTCN(h=a.hidden, drop=a.dropout).to(dev)
    else:
        model = (CausalTCN(d_in, h=a.hidden, drop=a.dropout) if a.model == 'tcn' else CausalGRU(d_in, h=a.hidden, drop=a.dropout)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=a.wd)
    steps = a.epochs*math.ceil(len(tr_idx)/a.batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.05)
    run = f'{a.model}_s{a.seed}{a.tag}'; out = HERE/'runs'/run; out.mkdir(parents=True, exist_ok=True)
    best, log, t0 = -1, [], time.time()
    for ep in range(a.epochs):
        model.train(); order = rng.permutation(tr_idx); tot = 0
        for k in range(0, len(order), a.batch):
            base, face = batch_crops(tr, order[k:k+a.batch], a.crop, rng)
            xy, sc, y, mk = [t.to(dev) for t in base]; face = [t.to(dev) for t in face] if face else None
            xy, sc, face = augment(xy, sc, rng, face)
            lo = ((F.binary_cross_entropy_with_logits(model(features(xy, sc, stats, face)), y, pos_weight=pos_w, reduction='none')*cls_w).sum(-1)/cls_w.sum()*mk).sum()/mk.sum()
            opt.zero_grad(); lo.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); tot += lo.item()
        if (ep+1) % 5 == 0 or ep == a.epochs-1:
            P, Y, Fp = predict(model, tr, stats, dev, dev_idx); r = evaluate(P, Y, Fp, names, eval_classes)
            log.append({'epoch': ep+1, 'train_loss': tot/math.ceil(len(order)/a.batch), 'dev_mAP': r['mAP_eval_classes'], 'min': (time.time()-t0)/60})
            print(json.dumps(log[-1]), flush=True)
            if r['mAP_eval_classes'] > best:
                best = r['mAP_eval_classes']; torch.save(model.state_dict(), out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt', weights_only=True))
    P, Y, Fp = predict(model, tr, stats, dev, dev_idx); th = best_thresholds(P, Y)
    dev_res = evaluate(P, Y, Fp, names, eval_classes, th)
    P, Y, Fp = predict(model, va, stats, dev); val_res = evaluate(P, Y, Fp, names, eval_classes, th)
    res = dict(args=vars(a), params=sum(p.numel() for p in model.parameters()), receptive_frames=model.receptive,
               eval_classes=[names[c] for c in eval_classes], train_minutes=(time.time()-t0)/60, log=log,
               dev=dev_res, val=val_res, thresholds=th,
               notes=('dev = appearance-clustered subject groups held out from train (~12% videos)' if a.dev_split == 'group'
                      else 'dev = random 10% of train videos (shares subjects with train)') + '; val = official, subject-disjoint')
    (out/'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{run}: dev mAP {dev_res['mAP_eval_classes']:.3f}, VAL mAP {val_res['mAP_eval_classes']:.3f} "
          f"(all {val_res['mAP_all_with_positives']:.3f}), {res['train_minutes']:.1f} min", flush=True)


if __name__ == '__main__':
    main()
