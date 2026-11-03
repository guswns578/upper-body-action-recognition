"""Draw cached COCO-WholeBody face landmarks (indices 23..90)."""
import cv2
import numpy as np

CHAINS = [list(range(0,17)), list(range(17,22)), list(range(22,27)),
          list(range(27,31)), list(range(31,36)),
          [36,37,38,39,40,41,36], [42,43,44,45,46,47,42],
          list(range(48,60))+[48], list(range(60,68))+[60]]

def draw_face(frame, person, scale=1.0, threshold=.5):
    if not person: return 0
    xy=np.asarray(person['keypoints_xy'][23:91])*scale
    scores=np.asarray(person['raw_scores'][23:91])
    h,w=frame.shape[:2]
    good=(scores>=threshold)&np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    for chain in CHAINS:
        for a,b in zip(chain,chain[1:]):
            if good[a] and good[b]:
                cv2.line(frame,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),(255,220,70),1,cv2.LINE_AA)
    for pt in xy[good]:cv2.circle(frame,tuple(pt.astype(int)),2,(255,220,70),-1,cv2.LINE_AA)
    return int(good.sum())

if __name__=='__main__':
    from pathlib import Path
    import json,subprocess
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    root=Path(__file__).resolve().parents[2]/'src/public_video_test'
    out=root/'gesture_result'
    records=[json.loads(s) for s in (root/'hands_result/wholebody_keypoints.jsonl').read_text().splitlines()]
    cap=cv2.VideoCapture(str(out/'gesture_descriptions_ko.mp4'));assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
    encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s','960x940','-r',str(fps),'-i','-',
        '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
        str(out/'gesture_descriptions_face_ko.mp4')],stdin=subprocess.PIPE)
    counts=[]
    for i,rec in enumerate(records):
        ok,frame=cap.read();assert ok
        n=draw_face(frame[:720],rec.get('person'),scale=1.5);counts.append(n)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));d=ImageDraw.Draw(im)
        d.rectangle((0,843,959,873),fill='black')
        d.text((18,847),'몸·얼굴 좌표: RTMW / 손 좌표·분류: MediaPipe',font=font,fill='#bfcfff')
        d.rectangle((10,10,350,40),fill='black')
        d.text((18,12),f'얼굴 표시: {n}/68점 (점수 ≥ 0.5)',font=font,fill='#46dcff')
        encoder.stdin.write(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR).tobytes())
        if i in [74,299,500]:im.save(out/f'face_check_{i:04d}.jpg')
    assert not cap.read()[0]
    cap.release();encoder.stdin.close();assert encoder.wait()==0
    check=cv2.VideoCapture(str(out/'gesture_descriptions_face_ko.mp4'));decoded=0
    while check.read()[0]:decoded+=1
    check.release();assert decoded==len(records)
    result={'frames':decoded,'face_indices':[23,90],'threshold':.5,'displayed_points_min':min(counts),
        'displayed_points_mean':sum(counts)/len(counts),'displayed_points_max':max(counts),
        'source':'cached RTMW coordinates; original gesture captions retained','accuracy_evaluated':False}
    (out/'face_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)
