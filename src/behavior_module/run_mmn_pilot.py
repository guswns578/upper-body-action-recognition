"""Official MMN iMiGUE joint model pilot; approximate cross-estimator adapter.

22-point order follows official feeder bones (OpenPose upper body + 5 tips/hand).
Finger-tip choice is inferred from same-author processing reference; not validated
against MMN training annotations. This experiment is not a benchmark reproduction.
"""
from pathlib import Path
import sys,json,time,hashlib,importlib.util
import numpy as np
import torch,yaml,cv2

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'src/micro_action_pilot';OUT.mkdir(exist_ok=True)
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

LABELS=['목을 움츠린다','볼을 부풀리거나 크게 숨 쉬는 동작','모자를 만진다','머리를 만지거나 긁는다',
        '이마를 만지거나 긁는다','얼굴을 가린다','눈을 비빈다','얼굴 부위를 만지거나 긁는다','귀를 만진다',
        '손톱을 문다','턱을 만진다','목을 만지거나 긁는다','머리카락을 만지거나 정리한다',
        '단추·옷깃·넥타이를 만진다','목 아래 쇄골 사이에 손을 댄다','등을 긁는다','팔짱을 낀다',
        '옷의 먼지를 턴다','팔을 몸 뒤로 둔다','몸통을 움직인다','상체를 곧게 세워 앉는다',
        '팔을 만지거나 긁는다','두 손을 비비거나 잡는다','손가락을 깍지 낀다','양손 손끝을 맞댄다',
        '물체를 만지거나 조작한다','팔을 뒤로 당긴다','고개를 든다','입술을 누르거나 다문다',
        '손을 허리에 둔다','어깨를 흔든다','설명에 수반되는 몸짓 (포괄 범주)']

def convert(person):
    if not person:return np.zeros((22,2),np.float32)
    p=np.asarray(person['keypoints_xy'],np.float32)
    # OpenPose upper body 0..7 then R/L eye and ear. Anatomical sides, no mirroring.
    torso=np.stack([p[0],(p[5]+p[6])/2,p[6],p[8],p[10],p[5],p[7],p[9],p[2],p[1],p[4],p[3]])
    return np.concatenate([torso,p[[116,120,124,128,132]],p[[95,99,103,107,111]]])

def prepare(value,feeder):
    # Execute the unmodified upstream __getitem__ without loading dataset pickles.
    obj=object.__new__(feeder.Feeder);obj.split='test';obj.repeat=1;obj.time_steps=64;obj.partition=True
    obj.data_type='j';obj.label=[0];obj.data=[value];obj.data_dict=[{'label':0}]
    obj.new_idx=np.array([4,12,13,14,15,16,7,17,18,19,20,21,0,8,9,10,11,1,2,5,3,6])
    data,t,_,_=obj[0];return torch.from_numpy(data.astype(np.float32))[None],torch.from_numpy(t)[None]

def main():
    torch.set_num_threads(4)
    mod=load_module('official_mmn',ROOT/'work/MMN/model/MMN.py')
    feeder=load_module('official_feeder',ROOT/'work/MMN/feeders/feeder_imigue.py')
    cfg=yaml.safe_load((ROOT/'work/MMN/config/test/iMiGUE_J.yaml').read_text())
    model=mod.MMN_(**cfg['model_args']);checkpoint=ROOT/'work/pose_models/MMN_iMiGUE_J.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
    summaries={}
    for name,poses,video in [('ipn','src/public_video_test/hands_result/wholebody_keypoints.jsonl','src/public_video_test/ipn_217_first20s.mp4'),
                             ('broadcast','src/behavior_module/results_hands/wholebody_keypoints.jsonl','src/pose_test/new_motion_video/test_35s_52s.mp4')]:
        recs=[json.loads(s) for s in (ROOT/poses).read_text().splitlines()]
        value=np.stack([convert(r.get('person')) for r in recs]);cap=cv2.VideoCapture(str(ROOT/video));fps=cap.get(5);cap.release()
        results=[]
        with torch.inference_mode():
            for end in range(99,len(recs),15):
                start=end-99;t0=time.perf_counter();x,t=prepare(value[start:end+1],feeder);p=model(x,t).softmax(-1)[0].numpy();elapsed=(time.perf_counter()-t0)*1000
                ids=np.argsort(p)[::-1][:5];assert np.isfinite(p).all() and np.isclose(p.sum(),1,atol=1e-5)
                results.append(dict(emitted_frame=end,window_start_s=start/fps,window_end_s=end/fps,
                    source_time_s=recs[end]['source_time_s'],top5=[dict(class_id_zero_based=int(k),description_ko=LABELS[k],score=float(p[k])) for k in ids],
                    probabilities=p.tolist(),inference_ms=elapsed,adapter_validated_on_training_data=False,
                    status='diagnostic_candidate_only',sampled_frames=(start+np.linspace(0,99,64).astype(int)).tolist()))
        (OUT/f'{name}_mmn_predictions.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in results)+'\n',encoding='utf-8')
        from collections import Counter
        summaries[name]={'windows':len(results),'top1_counts':dict(Counter(r['top5'][0]['description_ko'] for r in results)),
            'mean_inference_ms_excluding_first':float(np.mean([r['inference_ms'] for r in results[1:]]))}
    summary=dict(model='MMN iMiGUE J',strict_state_keys=len(model.state_dict()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_drive_id='1-C6-4kENj1oFEyx60JLWGXzbIAWARXQ3',source_commit=(ROOT/'work/MMN/.git/refs/heads/main').read_text().strip(),
        source_model_sha256=hashlib.sha256((ROOT/'work/MMN/model/MMN.py').read_bytes()).hexdigest(),results=summaries,
        labels_source='original iMiGUE Fig2, same-author notebook label-1/99->31; checkpoint class order not independently validated',
        limitations=['RTMW to OpenPose 22-point semantic adapter unvalidated against training data',
            'hidden hands can have false coordinates; no confidence channels in this checkpoint',
            'model predicts 32 fixed labels; illustrative-body-language is broad, not fine hand actions',
            'no identity tracking or scene-cut reset; no ground truth accuracy claim'])
    (OUT/'mmn_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
