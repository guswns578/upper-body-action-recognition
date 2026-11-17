"""(gpu_env) 학습된 모델을 '실제로 얼마나 맞고 틀리나'로 평가한다(공식 val, 처음 보는 37명).

- 임계값: 개발셋(사람 군집 단위)에서 클래스별 F1 최대가 되는 값(results.json의 thresholds) — val로 고르지 않는다.
  실행을 여러 개 주면 확률을 평균한 앙상블로 평가하고, 임계값은 앙상블의 개발셋 출력으로 다시 고른다.
- 프레임 단위: 정밀도(켜진 것 중 맞은 비율), 재현율(실제 행동 중 잡은 비율), F1.
- 구간 단위: 정답 구간과 예측 구간이 조금이라도 겹치면(tIoU>0) 검출로 본다. 구간 검출률, 예측 구간 중 정답과 안 겹친 비율(헛검출),
  분당 헛검출 수. 예측 구간은 임계값을 넘는 연속 프레임(최소 길이·평활 없음).

    python practical_eval.py tcn2_s0_grp_face
    python practical_eval.py tcn_s0_grp_nohead_hands tcn_s1_grp_nohead_hands tcn_s2_grp_nohead_hands --out ens_nohead_hands
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from train import (
    Split,
    TwoStreamTCN,
    CausalTCN,
    CausalGRU,
    predict,
    segments,
    average_precision,
    best_thresholds,
    CACHE,
    HERE,
)


def load(run, dev):
    r = json.loads((HERE / "runs" / run / "results.json").read_text(encoding="utf-8"))
    a = r["args"]
    face, hands = a.get("face", False), a.get("hands", False)
    d_in = 5 * (15 if hands else 13) + (89 if face else 0)
    m = (
        TwoStreamTCN(h=a["hidden"], drop=a["dropout"])
        if a["model"] == "tcn2"
        else (
            CausalTCN(d_in, h=a["hidden"], drop=a["dropout"])
            if a["model"] == "tcn"
            else CausalGRU(d_in, h=a["hidden"], drop=a["dropout"])
        )
    )
    m.load_state_dict(torch.load(HERE / "runs" / run / "best.pt", weights_only=True))
    return m.to(dev), r, (face, hands)


def dev_indices(tr):
    """train.py와 같은 군집 단위 개발셋."""
    grp = json.loads(Path("data/MMA-52/subject_groups/train_groups.json").read_text())
    g = np.array([grp[v] for v in tr.videos])
    held = []
    for k in np.random.default_rng(12345).permutation(np.unique(g)):
        held.append(k)
        if np.isin(g, held).sum() >= len(tr) * 0.12:
            break
    return np.nonzero(np.isin(g, held))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    models = [load(r, dev) for r in a.runs]
    cfg = models[0][2]
    r0 = models[0][1]
    assert all(m[2] == cfg for m in models), "같은 입력 설정의 실행만 묶는다"
    tr, va = Split("train", *cfg), Split("val", *cfg)
    stats = torch.tensor([tr.score.mean(), tr.score.std()], device=dev)
    names = json.loads((CACHE / "meta.json").read_text())["names"]

    def ens(split, idx=None):
        outs = [predict(m, split, stats, dev, idx) for m, _, _ in models]
        return [np.mean([o[0][k] for o in outs], 0) for k in range(len(outs[0][0]))], outs[0][1]

    if len(models) == 1:
        th = r0["thresholds"]
        P, Y = ens(va)
    else:
        Pd, Yd = ens(tr, dev_indices(tr))
        th = best_thresholds(Pd, Yd)
        P, Y = ens(va)
    minutes = sum(len(y) for y in Y) / 30 / 60
    p_all, y_all = np.concatenate(P), np.concatenate(Y)
    rows = {}
    for n in r0["eval_classes"]:
        c = names.index(n)
        tp = fp = fn = 0
        seg_hit = seg_tot = pred_tot = pred_false = 0
        for p, y in zip(P, Y):
            q = p[:, c] > th[c]
            g = y[:, c].astype(bool)
            tp += (q & g).sum()
            fp += (q & ~g).sum()
            fn += (~q & g).sum()
            gs, ps = segments(g), segments(q)
            seg_tot += len(gs)
            seg_hit += sum(any(ps_s <= ge and ps_e >= gs_ for ps_s, ps_e in ps) for gs_, ge in gs)
            pred_tot += len(ps)
            pred_false += sum(
                not any(ps_s <= ge and ps_e >= gs_ for gs_, ge in gs) for ps_s, ps_e in ps
            )
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        rows[n] = dict(
            AP=average_precision(y_all[:, c].astype(np.float32), p_all[:, c]),
            threshold=th[c],
            precision=float(prec),
            recall=float(rec),
            F1=float(2 * prec * rec / max(1e-9, prec + rec)),
            segment_detect=seg_hit / max(1, seg_tot),
            segments=seg_tot,
            false_segments_frac=pred_false / max(1, pred_tot),
            false_segments_per_min=pred_false / minutes,
        )
    out = HERE / "runs" / (a.out or a.runs[0])
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(
        runs=a.runs,
        mAP=float(np.nanmean([v["AP"] for v in rows.values()])),
        mean_F1=float(np.mean([v["F1"] for v in rows.values()])),
        false_segments_per_min_total=float(sum(v["false_segments_per_min"] for v in rows.values())),
    )
    (out / "practical_val.json").write_text(
        json.dumps(dict(summary=summary, classes=rows), indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f'{"class":32s}  AP    prec  recall  F1   seg_detect  false_seg%  false/min')
    for n, v in sorted(rows.items(), key=lambda kv: -kv[1]["AP"]):
        print(
            f"{n[:32]:32s} {v['AP']:.2f}  {v['precision']:.2f}  {v['recall']:.2f}   {v['F1']:.2f}   {v['segment_detect']:.2f}({v['segments']:3d})  {v['false_segments_frac']:.2f}       {v['false_segments_per_min']:.1f}"
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
