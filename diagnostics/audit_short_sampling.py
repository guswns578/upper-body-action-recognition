from pathlib import Path
import sys,json,pickle
import numpy as np,torch
R=Path(__file__).resolve().parents[1];O=R/'src/cause_isolation'
sys.path.insert(0,str(R/'src/behavior_module'));from action_adapter import ActionModel
torch.set_num_threads(3);m=ActionModel(R/'work/pose_models/stgcnpp_ntu120_2d_j.pth')
with (O/'selected_ntu.pkl').open('rb') as f:data=pickle.load(f)
rows=[]
for a in data:
    n=max(1,a['keypoint'].shape[1]//4);x=a['keypoint'][:,-n:];s=a['keypoint_score'][:,-n:]
    row=dict(id=a['frame_dir'],label=int(a['label']),source_frames=n,variants={})
    for mode in ['official_loop','monotonic_repeat','native_length']:
        if mode=='official_loop':p,inds=m.predict(x,s,a['img_shape'])
        elif mode=='monotonic_repeat':
            inds=np.linspace(0,n-1,100).round().astype(int);p,_=m.predict(x[:,inds],s[:,inds],a['img_shape']);inds=inds.tolist()
        else:
            r=dict(keypoint=x.copy(),keypoint_score=s.copy(),img_shape=a['img_shape'])
            r=m.official['PreNormalize2D']()(r);r=m.official['FormatGCNInput'](num_person=2)(r)
            with torch.inference_mode():p=m.cls_head(m.backbone(torch.from_numpy(r['keypoint']))).softmax(-1).mean(0).numpy()
            inds=list(range(n))
        row['variants'][mode]=dict(top1=int(p.argmax()),correct=bool(p.argmax()==a['label']),wraps=int((np.diff(inds)<0).sum()),indices=inds)
    rows.append(row)
(O/'short_sampling.json').write_text(json.dumps(rows,indent=2));print({mode:sum(r['variants'][mode]['correct'] for r in rows) for mode in rows[0]['variants']})
