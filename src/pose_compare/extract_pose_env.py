"""(pose_env) RTMPose-m, RTMW-l, MediaPipe Pose heavy(IMAGE / VIDEO 모드) 추출.

- RTMPose/RTMW: rtmlib 기본 방식대로 원래 박스를 넘긴다(내부에서 1.25배 확장).
- MediaPipe: 1.25배 확장 크롭을 넘기고 좌표를 원본으로 되돌린다.
  IMAGE = 프레임 독립 추론, VIDEO = 공식 스트리밍 모드(이전 프레임 추적·평활 포함).
"""

import sys, time
import numpy as np
import cv2
import onnxruntime as ort
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from rtmlib import RTMPose
from common import ROOT, CLIPS, MAP, SOURCES, read_frames, expand, save_raw

M = ROOT / "work/pose_models"


def ort_session(model):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 4
    opt.inter_op_num_threads = 1
    model.session = ort.InferenceSession(
        model.onnx_model, sess_options=opt, providers=["CPUExecutionProvider"]
    )
    return model


def mp_landmarker(mode):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(
                model_asset_path=str(M / "mediapipe/pose_landmarker_heavy.task")
            ),
            running_mode=mode,
            num_poses=1,
        )
    )


def main():
    only = set(sys.argv[1:])
    rtm = ort_session(
        RTMPose(
            str(M / "rtmpose_m.onnx"),
            model_input_size=(192, 256),
            backend="onnxruntime",
            device="cpu",
            to_openpose=False,
        )
    )
    rtw = ort_session(
        RTMPose(
            str(M / "rtmw_384.onnx"),
            model_input_size=(288, 384),
            backend="onnxruntime",
            device="cpu",
            to_openpose=False,
        )
    )
    idx17, idx33 = MAP["coco17"], MAP["mediapipe33"]
    for name, *_ in SOURCES:
        if only and name not in only:
            continue
        c = np.load(CLIPS / f"{name}.npz")
        boxes = c["boxes"]
        fps = float(c["fps"])
        W, H = c["size"]
        T = len(boxes)
        out = {
            k: (np.full((T, 13, 2), np.nan), np.full((T, 13), np.nan), np.zeros(T))
            for k in ["rtmpose_m", "rtmw_l", "mediapipe_image", "mediapipe_video"]
        }
        hands = np.full((T, 42, 2), np.nan)
        hand_sc = np.full((T, 42), np.nan)
        mpi, mpv = mp_landmarker(vision.RunningMode.IMAGE), mp_landmarker(vision.RunningMode.VIDEO)
        for t, _, f in read_frames(
            c["path"].item(),
            float(c["start_s"]),
            None if T == 0 else (float(c["start_s"]) + T / fps + 1e-6),
        ):
            if t >= T:
                break
            b = boxes[t]
            if np.isnan(b[0]):
                continue
            for key, model in [("rtmpose_m", rtm), ("rtmw_l", rtw)]:
                t0 = time.perf_counter()
                kp, sc = model(f, bboxes=[b.tolist()])
                out[key][2][t] = (time.perf_counter() - t0) * 1e3
                out[key][0][t] = kp[0][idx17]
                out[key][1][t] = sc[0][idx17]
                if key == "rtmw_l":
                    hands[t] = kp[0][91:133]
                    hand_sc[t] = sc[0][91:133]
            x1, y1, x2, y2 = expand(b, W, H).astype(int)
            crop = np.ascontiguousarray(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop)
            for key, lm, call in [
                ("mediapipe_image", mpi, lambda: mpi.detect(img)),
                (
                    "mediapipe_video",
                    mpv,
                    lambda: mpv.detect_for_video(img, int(round(t * 1000 / fps))),
                ),
            ]:
                t0 = time.perf_counter()
                r = call()
                out[key][2][t] = (time.perf_counter() - t0) * 1e3
                if r.pose_landmarks:
                    L = r.pose_landmarks[0]
                    out[key][0][t] = [
                        [L[i].x * (x2 - x1) + x1, L[i].y * (y2 - y1) + y1] for i in idx33
                    ]
                    out[key][1][t] = [L[i].visibility for i in idx33]
        mpi.close()
        mpv.close()
        for key, (xy, sc, ms) in out.items():
            extra = (
                {"hands_xy": hands.astype(np.float32), "hands_score": hand_sc.astype(np.float32)}
                if key == "rtmw_l"
                else None
            )
            save_raw(key, name, xy, sc, ms, extra)
        print(name, {k: f"{np.median(v[2][v[2] > 0]):.1f}ms" for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
