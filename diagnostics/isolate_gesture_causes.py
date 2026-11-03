"""Paired diagnostic on held-out NTU samples; not an official full benchmark."""
from pathlib import Path
import json,pickle,sys,time,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[1];O=R/'src/cause_isolation';O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'src/behavior_module'))
from action_adapter import ActionModel
torch.set_num_threads(4)
labels=(R/'work/pyskl/tools/data/label_map/nturgbd_120.txt').read_text(encoding='utf-8').splitlines()
print('loading official annotations',flush=True)
with (R/'work/audit_data/ntu120_hrnet.pkl').open('rb') as f:d=pickle.load(f)
val=set(d['split']['xsub_val']);rng=np.random.RandomState(20261008)
selected=[]
for label in [22,30,68,92,94,95]:
    candidates=sorted([a for a in d['annotations'] if a['frame_dir'] in val and a['label']==label],key=lambda a:a['frame_dir'])
    for i in rng.choice(len(candidates),5,replace=False):selected.append(candidates[i])
del d
print('selected',len(selected),flush=True)
model=ActionModel(R/'work/pose_models/stgcnpp_ntu120_2d_j.pth')
variants=['original_10clip','original_1clip','upper_only_10clip','zoom2_10clip','last_quarter_10clip','static_middle_10clip']
rows=[]
for a in selected:
    xy=a['keypoint'].astype(np.float32);score=a['keypoint_score'].astype(np.float32);h,w=a['img_shape']
    row=dict(id=a['frame_dir'],label=int(a['label']),label_name=labels[a['label']],frames=int(a['total_frames']),shape=[int(h),int(w)],variants={})
    for v in variants:
        xx=xy.copy();ss=score.copy();nc=1 if v=='original_1clip' else 10
        if v=='upper_only_10clip':xx[:,:,11:]=0;ss[:,:,11:]=0
        if v=='zoom2_10clip':
            visible=ss>.01
            xx=(xx-np.array([w/2,h/2],dtype=np.float32))*2+np.array([w/2,h/2],dtype=np.float32)
            xx[~visible]=0
        if v=='last_quarter_10clip':xx=xx[:,-max(1,xx.shape[1]//4):];ss=ss[:,-xx.shape[1]:]
        if v=='static_middle_10clip':
            m=xx.shape[1]//2;xx=np.repeat(xx[:,m:m+1],xx.shape[1],axis=1);ss=np.repeat(ss[:,m:m+1],ss.shape[1],axis=1)
        t=time.perf_counter();p,inds=model.predict(xx,ss,(h,w),num_clips=nc);top=np.argsort(p)[-5:][::-1]
        row['variants'][v]=dict(top1=int(top[0]),correct=bool(top[0]==a['label']),target_probability=float(p[a['label']]),
            target_rank=int(np.where(np.argsort(p)[::-1]==a['label'])[0][0]+1),top5=[dict(id=int(k),label=labels[k],score=float(p[k])) for k in top],seconds=time.perf_counter()-t)
    rows.append(row);(O/'ntu_paired_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(a['frame_dir'],{v:r['top1'] for v,r in row['variants'].items()},flush=True)
summary={}
for v in variants:
    correct=sum(r['variants'][v]['correct'] for r in rows)
    lost=sum(r['variants']['original_10clip']['correct'] and not r['variants'][v]['correct'] for r in rows)
    gained=sum(not r['variants']['original_10clip']['correct'] and r['variants'][v]['correct'] for r in rows)
    summary[v]=dict(correct=correct,n=len(rows),lost_from_original=lost,gained_from_original=gained,
        by_class={labels[k]:sum(r['variants'][v]['correct'] for r in rows if r['label']==k) for k in [22,30,68,92,94,95]})
(O/'ntu_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
with (O/'selected_ntu.pkl').open('wb') as f:pickle.dump(selected,f)
