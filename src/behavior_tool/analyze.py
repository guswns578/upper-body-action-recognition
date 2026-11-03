"""Generic local multi-person video analysis. No video-specific/manual labels."""
from pathlib import Path
import sys,argparse,json,time,datetime,traceback,warnings,subprocess
from collections import Counter
import numpy as np
import cv2,onnxruntime as ort
import supervision as sv
from rtmlib import RTMPose,YOLOX
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from core import GeometryQuality,ArmsCrossedRules,CutDetector,Window,upper_body_quality,assign_hands,make_mmn_points,HAND_LABELS

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src/behavior_module'))
from face_overlay import draw_face

class MMNBackend:
    def __init__(self,models):
        import torch,yaml
        from run_mmn_pilot import load_module,prepare,LABELS
        self.torch=torch;self.prepare=prepare;self.labels=LABELS;torch.set_num_threads(4)
        module=load_module('tool_mmn',ROOT/'work/MMN/model/MMN.py')
        self.feeder=load_module('tool_feeder',ROOT/'work/MMN/feeders/feeder_imigue.py')
        cfg=yaml.safe_load((ROOT/'work/MMN/config/test/iMiGUE_J.yaml').read_text())
        self.model=module.MMN_(**cfg['model_args'])
        self.model.load_state_dict(torch.load(models/'MMN_iMiGUE_J.pt',map_location='cpu',weights_only=True),strict=True);self.model.eval()
    def predict(self,items):
        values=np.stack([p for _,p in items]);x,t=self.prepare(values,self.feeder)
        with self.torch.inference_mode():prob=self.model(x,t).softmax(-1)[0].numpy()
        order=np.argsort(prob)[::-1][:3]
        return [{'class_id':int(k),'description_ko':self.labels[k],'score':float(prob[k])} for k in order]

def detector_result(model,frame):
    image,ratio=model.preprocess(frame);outputs=model.inference(image)
    det=np.asarray(outputs[0])[0];labels=np.asarray(outputs[1]).reshape(-1)
    if det.shape[-1]!=5:raise RuntimeError('This detector adapter requires YOLOX ONNX dets [1,N,5] plus labels')
    good=(labels==0)&(det[:,4]>=.1)&np.isfinite(det).all(1)
    boxes=det[good,:4].copy()/ratio;scores=det[good,4].copy()
    h,w=frame.shape[:2];boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w-1);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h-1)
    good=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    return sv.Detections(xyxy=boxes[good].astype(np.float32),confidence=scores[good].astype(np.float32),class_id=np.zeros(int(good.sum()),dtype=int))

def create_models(directory,hands=False):
    det=YOLOX(str(directory/'yolox_tiny.onnx'),model_input_size=(416,416),backend='onnxruntime',device='cpu')
    pose=RTMPose(str(directory/('rtmw_384.onnx' if hands else 'rtmpose_m.onnx')),model_input_size=(288,384) if hands else (192,256),backend='onnxruntime',device='cpu',to_openpose=False)
    for model in [det,pose]:
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        model.session=ort.InferenceSession(model.onnx_model,sess_options=opt,providers=['CPUExecutionProvider'])
    recognizer=vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=python.BaseOptions(model_asset_path=str(directory/'gesture_recognizer.task')),
        running_mode=vision.RunningMode.IMAGE,num_hands=2)) if hands else None
    return det,pose,recognizer

def detect_hands(recognizer,frame,box,xy):
    x1,y1,x2,y2=map(int,box);crop=frame[y1:y2,x1:x2]
    result=recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))
    points=[np.array([[p.x*(x2-x1)+x1,p.y*(y2-y1)+y1] for p in hand]) for hand in result.hand_landmarks]
    association=assign_hands(xy,points,box);hands=[]
    for side,index in association.items():
        cats=result.gestures[index];top=max(cats,key=lambda c:c.score) if cats else None
        status='candidate' if top and top.category_name in HAND_LABELS and top.score>=.5 else 'unclassified'
        hands.append({'side':side,'landmarks_xy':points[index].tolist(),'status':status,
            'label':top.category_name if top else None,'score':float(top.score) if top else None,
            'description_ko':HAND_LABELS[top.category_name] if status=='candidate' else '손 모양 판단 보류',
            'association':'nearest_body_wrist_geometry_not_verified_identity'})
    return hands,len(points)

def draw_person(frame,p):
    box=p['bbox_xyxy'];a,b,c,d=map(int,box);cv2.rectangle(frame,(a,b),(c,d),(70,220,255),2)
    cv2.putText(frame,p['track_id'],(a,max(18,b-7)),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,220,255),2)
    if 'pose' not in p:return
    xy=np.asarray(p['pose']['keypoints_xy']);sc=np.asarray(p['pose']['raw_scores']);h,w=frame.shape[:2]
    _,good=upper_body_quality(xy,sc,w,h)
    for j,k in [(5,6),(5,7),(7,9),(6,8),(8,10)]:
        if good[j] and good[k]:cv2.line(frame,tuple(xy[j].astype(int)),tuple(xy[k].astype(int)),(70,240,90),2)
    if len(xy)==133:draw_face(frame,p['pose'])
    for hand in p['hands']:
        pts=np.asarray(hand['landmarks_xy']).astype(int)
        for finger in range(5):
            chain=[0]+list(range(1+finger*4,5+finger*4))
            for j,k in zip(chain,chain[1:]):cv2.line(frame,tuple(pts[j]),tuple(pts[k]),(255,140,80),2)

def analyze(args):
    video=args.video.resolve()
    if not video.is_file():raise ValueError(f'영상 파일 없음: {video}')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('영상 형식/코덱을 읽을 수 없습니다.')
    fps=cap.get(cv2.CAP_PROP_FPS);expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps<=0:cap.release();raise ValueError('영상 FPS를 확인할 수 없습니다.')
    out=args.output or ROOT/'src/video_tool_runs'/f'{video.stem[:40]}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}'
    if out.exists() and any(out.iterdir()):cap.release();raise ValueError('출력 폴더에 파일이 있습니다. 새 폴더를 지정해 주세요.')
    out.mkdir(parents=True,exist_ok=True)
    summary={'status':'running','input':str(video),'output':str(out.resolve()),'fps':fps,'expected_frames':expected,
        'action_backend':args.action_model,'manual_annotations_used':False,'identity_verified':False,
        'limitations':['ByteTrack IDs may switch during overlap; no cross-shot identity matching',
            'hand ownership uses geometry and may be ambiguous; missing hands are not invented',
            'MMN coordinate adapter and labels remain diagnostic, not calibrated action accuracy',
            'hard cuts detected heuristically; fades or similar shots may be missed',
            'review video is silent CFR; JSON retains decoded timestamps where available',
            'CPU sequential analysis is not a live realtime guarantee'],
        'config':{'window_seconds':args.window,'min_person_height':args.min_person_height,'cut_threshold':args.cut_threshold,'track_selection':args.track_id,'hands_enabled':args.hands,'pose_scores_gate_coordinates':False}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    recognizer=None;encoder=None;raw=None;events=None;started=time.perf_counter();stats=Counter();track_ids=set();scene=0;count=0
    try:
        det,pose,recognizer=create_models(args.models,args.hands)
        action=MMNBackend(args.models) if args.action_model=='mmn' else None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning)
            tracker=sv.ByteTrack(track_activation_threshold=.3,lost_track_buffer=15,frame_rate=max(1,round(fps)),minimum_consecutive_frames=2)
        # Offline cut prepass enables censoring BEFORE a cut, too.
        cut_frames=[];scanner=CutDetector(args.cut_threshold);scan_i=0
        while args.max_frames is None or scan_i<args.max_frames:
            scan_ok,scan_frame=cap.read()
            if not scan_ok:break
            detected,_=scanner.update(scan_frame)
            if detected:cut_frames.append(scan_i)
            scan_i+=1
        cap.release();cap=cv2.VideoCapture(str(video))
        summary['cut_frames']=cut_frames
        mapping=json.loads(args.speaker_map.read_text(encoding='utf-8')) if args.speaker_map else {}
        cuts=CutDetector(args.cut_threshold);windows={};last_seen={};latest={};previous_time=-1.
        qualities={};rules={}
        def emit_segments(tid,items):
            for item in items:
                item.update(track_id=tid,speaker_id=mapping.get(tid),target_pid=None,model='arms_crossed_rules_v1',visibility='unknown')
                events.write(json.dumps(item,ensure_ascii=False)+'\n')
                stats['action_updates']+=1
        raw=(out/'observations.jsonl').open('w',encoding='utf-8');events=(out/'action_updates.jsonl').open('w',encoding='utf-8')
        font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
        if not args.no_video:
            encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
                '-f','rawvideo','-pix_fmt','bgr24','-s','960x900','-r',str(fps),'-i','-',
                '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'review_ko.mp4')],stdin=subprocess.PIPE)
        while args.max_frames is None or count<args.max_frames:
            ok,frame=cap.read()
            if not ok:break
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            if not np.isfinite(pts) or (count>0 and pts<=previous_time):pts=max(count/fps,previous_time+1/fps);stats['timestamp_fallback_frames']+=1
            previous_time=pts;h,w=frame.shape[:2]
            cut,cut_metrics=cuts.update(frame)
            boundary=any(abs(count-c)/fps<=.2 for c in cut_frames)
            if cut:
                
                for old_id,rule in rules.items():emit_segments(old_id,rule.finish(pts,'scene_cut'))
                rules.clear();qualities.clear()
                scene+=1;tracker.reset();windows.clear();last_seen.clear();latest.clear();stats['scene_cuts']+=1
            detections=detector_result(det,frame);tracked=tracker.update_with_detections(detections)
            people=[];seen=set()
            for box,score,local_id in zip(tracked.xyxy,tracked.confidence,tracked.tracker_id):
                tid=f'S{scene+1:03d}-P{int(local_id):04d}';track_ids.add(tid);seen.add(tid)
                person={'track_id':tid,'bbox_xyxy':box.tolist(),'detection_score':float(score),'hands':[],
                    'status':'unobservable','description_ko':'관찰 불가','speaker_id':mapping.get(tid),
                    'visibility':'unknown','quality_flags':[],'track_id_kind':'scene_local_bytetrack'}
                people.append(person)
                if args.track_id and tid not in args.track_id:person.update(status='not_selected',description_ko='선택 대상 아님');continue
                window=windows.setdefault(tid,Window(args.window,gap=max(.35,2/fps)))
                if pts-last_seen.get(tid,pts)>.15:window.clear();latest.pop(tid,None)
                last_seen[tid]=pts
                if box[3]-box[1]<args.min_person_height or box[2]-box[0]<32:
                    person['description_ko']='대상이 작아 세부 행동 관찰 보류';person['quality_flags']=['person_too_small'];window.clear();latest.pop(tid,None)
                    if tid in rules:emit_segments(tid,rules[tid].update(pts,None))
                    continue
                xy,sc=pose(frame,[box.tolist()]);xy=xy[0];sc=sc[0]
                person['pose']={'keypoints_xy':xy.tolist(),'raw_scores':sc.tolist(),'layout':'coco_wholebody_133' if args.hands else 'coco_17','score_is_probability':False}
                body_ok,good=upper_body_quality(xy,sc,w,h)
                body_ok,flags=qualities.setdefault(tid,GeometryQuality()).update(pts,xy,sc,w,h,box)
                if boundary:body_ok=False;flags.append("cut_boundary")
                person["quality_flags"]=flags
                hands,detected_hands=detect_hands(recognizer,frame,box,xy) if recognizer else ([],0)
                person['hands']=hands;person['detected_hand_count']=detected_hands
                person['quality']={'body_geometry_pass':body_ok,'associated_hands':len(hands),'visibility_verified':False}
                # Record detected, geometrically associated hands even when full-body action is unavailable.
                hand_text=' / '.join(('왼손' if hand['side']=='left' else '오른손')+': '+hand['description_ko'] for hand in hands)
                person['hand_description_ko']=(hand_text or '손 미검출 또는 인물·손목 연결 불확실') if args.hands else '손 추출 옵션 꺼짐'
                points=make_mmn_points(xy,{hand['side']:hand['landmarks_xy'] for hand in hands}) if body_ok and np.all(good[:5]) else None
                if args.action_model=='rules':
                    rule=rules.setdefault(tid,ArmsCrossedRules())
                    emit_segments(tid,rule.update(pts,xy if body_ok else None))
                    person.update(status='measurable' if body_ok else 'unobservable',
                        description_ko=('팔짱 후보' if rule.active else '지원 자세 미검출') if body_ok else '관찰 불가: 좌표 품질 또는 컷 경계')
                elif args.action_model=='none':
                    person.update(status='measurable' if body_ok else 'unobservable',description_ko='몸 좌표 기록 중' if body_ok else '관찰 불가')
                elif points is None:
                    window.clear();latest.pop(tid,None);person['description_ko']='상반신 행동 보류: 몸·양손 입력 부족'
                elif action is None:
                    person.update(status='action_disabled',description_ko='행동 모델 사용 안 함')
                else:
                    window.add(pts,points)
                    person.update(status='collecting',description_ko='행동 입력 수집 중')
                    if window.ready(pts):
                        candidates=action.predict(list(window.items));window.last_emit=pts
                        candidate_status='unverified_candidate' if candidates[0]['score']>=.5 else 'unclassified'
                        update={'track_id':tid,'emitted_time_s':pts,'window_start_s':window.items[0][0],'window_end_s':pts,
                            'status':candidate_status,'model':'MMN iMiGUE J','top3':candidates,'speaker_id':None,
                            'description_ko':('미검증 행동 후보: '+candidates[0]['description_ko']) if candidate_status=='unverified_candidate' else '지원 행동으로 판정하기 어려움',
                            'is_action_boundary':False,'adapter_validated':False}
                        latest[tid]=update;events.write(json.dumps(update,ensure_ascii=False)+'\n');stats['action_updates']+=1
                    if tid in latest:person.update(status=latest[tid]['status'],description_ko=latest[tid]['description_ko'],action=latest[tid])
                stats[person['status']]+=1
            for tid in list(windows):
                if tid not in seen:
                    if tid in rules:emit_segments(tid,rules[tid].update(pts,None))
                    if pts-last_seen.get(tid,0)>.15:windows[tid].clear();latest.pop(tid,None)
                    if pts-last_seen.get(tid,0)>2:windows.pop(tid,None);last_seen.pop(tid,None)
            record={'frame_index':count,'source_time_s':pts,'scene_id':scene+1,'scene_cut':cut,'cut_metrics':cut_metrics,
                'width':w,'height':h,'raw_person_detections':len(detections),'people':people}
            raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stats['person_observations']+=len(people)
            if encoder:
                for p in people:draw_person(frame,p)
                scale=min(960/w,540/h);rw,rh=round(w*scale),round(h*scale)
                im=Image.new('RGB',(960,900),(15,21,30));im.paste(Image.fromarray(cv2.cvtColor(cv2.resize(frame,(rw,rh)),cv2.COLOR_BGR2RGB)),((960-rw)//2,(540-rh)//2))
                d=ImageDraw.Draw(im);d.text((15,550),f'자동 분석 | {pts:.2f}초 | 장면 {scene+1} | 추적 인물 {len(people)}명',font=font,fill='white')
                visible=people[(count//max(1,round(fps*3))*4)%max(1,len(people)):][:4]
                for row,p in enumerate(visible):
                    d.text((15,588+row*64),p['track_id']+'  '+p['description_ko'],font=font,fill='#ffd18a')
                    d.text((15,618+row*64),p.get('hand_description_ko','세부 손 정보 없음'),font=small,fill='#bbdaff')
                if not people:d.text((15,600),'사람 미검출 또는 추적 확인 중',font=font,fill='#ffd18a')
                d.text((15,852),'모든 인물 결과는 JSON 저장 / 자막은 최대 4명씩 순환 / 감정·의도 추정 안 함',font=small,fill='#bbbbbb')
                d.text((15,878),'행동은 미검증 후보이며 점수는 정확도가 아닙니다. 가림·인물 겹침은 오류 가능.',font=small,fill='#bbbbbb')
                encoder.stdin.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR).tobytes())
                if count in [0,30,90]:im.save(out/f'preview_{count:05d}.jpg')
            count+=1
            if count%60==0:print(f'{count}/{expected} frames, {len(track_ids)} track IDs',flush=True)
        for tid,rule in rules.items():emit_segments(tid,rule.finish(previous_time,'end_of_input'))
        if count==0:raise RuntimeError('디코딩 가능한 영상 프레임이 없습니다.')
        if encoder:encoder.stdin.close();assert encoder.wait()==0;encoder=None
        summary.update(status='complete',processed_frames=count,scene_count=scene+1,track_ids=sorted(track_ids),statistics=dict(stats),
            processing_seconds=time.perf_counter()-started,stopped_at_max_frames=args.max_frames is not None and count>=args.max_frames)
    except BaseException as exc:
        summary.update(status='failed',processed_frames=count,error=str(exc));raise
    finally:
        cap.release()
        if raw:raw.close()
        if events:events.close()
        if recognizer:recognizer.close()
        if encoder:
            try:encoder.stdin.close()
            except OSError:pass
            encoder.wait()
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(out.resolve()),flush=True)

def main():
    ap=argparse.ArgumentParser(description='영상 → 다중 인물 좌표·손 모양·행동 후보 (로컬 실험 도구)')
    ap.add_argument('video',nargs='?',type=Path,help='없으면 영상 선택 창')
    ap.add_argument('--output',type=Path);ap.add_argument('--models',type=Path,default=ROOT/'work/pose_models')
    ap.add_argument('--action-model',choices=['rules','mmn','none'],default='rules')
    ap.add_argument('--hands',action='store_true',help='RTMW 133점과 MediaPipe 손 추출을 켭니다')
    ap.add_argument('--speaker-map',type=Path,help='장면별 track_id → speaker_id JSON 매핑')
    ap.add_argument('--window',type=float,default=2.5);ap.add_argument('--cut-threshold',type=float,default=.18)
    ap.add_argument('--min-person-height',type=int,default=96)
    ap.add_argument('--track-id',action='append',help='예: S001-P0001; 생략하면 전원')
    ap.add_argument('--max-frames',type=int);ap.add_argument('--no-video',action='store_true')
    args=ap.parse_args()
    if args.action_model=="mmn" and not args.hands:ap.error("MMN 실험 어댑터는 --hands가 필요합니다.")
    if args.window<=0 or args.min_person_height<=0 or (args.max_frames is not None and args.max_frames<=0):ap.error('수치 옵션은 양수여야 합니다.')
    if args.video is None:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk();root.withdraw()
        selected=filedialog.askopenfilename(title='분석할 영상 선택',filetypes=[('영상','*.mp4 *.avi *.mov *.mkv *.webm *.m4v'),('모든 파일','*.*')]);root.destroy()
        if not selected:return
        args.video=Path(selected)
    ort.set_default_logger_severity(3);cv2.setNumThreads(2)
    analyze(args)

if __name__=='__main__':main()
