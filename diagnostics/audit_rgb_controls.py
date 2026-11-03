from pathlib import Path
import sys,json,time
import numpy as np,torch,cv2,mmcv
from PIL import Image,ImageSequence
R=Path(__file__).resolve().parents[1];O=R/'src/cause_isolation';O.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'src/seated_references'));import compare_pretrained as cp
cp.OUT=O;torch.set_num_threads(3);cv2.setNumThreads(1);model=cp.FrozenRGB()
rows=[]
for name,label in [('folding-arms',16),('covering-face',5),('scratching-neck',11),('rubbing-eyes',6)]:
    path=R/'work/iMiGUE/pictures'/(name+'.gif');im=Image.open(path)
    raw=[np.array(f.convert('RGB')) for f in ImageSequence.Iterator(im)]
    frames=np.stack([mmcv.imresize(f,mmcv.rescale_size((f.shape[1],f.shape[0]),(float('inf'),224))) for f in raw])
    row=dict(name=name,target_id=label,target_name=cp.LABELS[label],frames=len(raw),source='Official iMiGUE README illustrative GIF; NOT held-out benchmark',variants={})
    for key,f,full in [('center_1clip',frames,False),('official_12view',frames,True),('static_middle_12view',np.repeat(frames[len(frames)//2:len(frames)//2+1],len(frames),axis=0),True)]:
        p,meta=model.predict(f,full);row['variants'][key]=dict(top1=int(p.argmax()),target_probability=float(p[label]),target_rank=int(np.where(np.argsort(p)[::-1]==label)[0][0]+1),top5=cp.packed(p,cp.LABELS)['top5'],sampling=meta)
        print(name,key,row['variants'][key]['top5'][0],flush=True)
    rows.append(row);(O/'rgb_controls.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
