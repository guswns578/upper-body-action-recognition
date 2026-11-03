# GPU 환경 구성 및 CPU/GPU 출력 동등성 검증 — 2026-10-26

## 환경

| 항목 | 기존 `work/pose_env` | 신규 `work/gpu_env` |
|---|---|---|
| Python | 3.12.14 (같은 기반 인터프리터) | 3.12.14 |
| torch / torchvision | 2.14.0+cpu / 0.29.0 | **2.14.0+cu130 / 0.29.0+cu130** |
| CUDA | 없음 | 13.0, RTX 5070 12GB, sm_120 지원 확인 |
| numpy / mmcv-lite / mmengine / opencv | 2.5.3 / 2.2.0 / 0.10.7 / 5.0.0.93 | 동일 버전 고정 |
| einops / timm / scipy / pyyaml | 0.8.2 / 1.0.29 / 1.18.1 / 6.0.3 | 동일 버전 고정 |

설치 명령(재현용):

```text
<기반 python> -m venv work/gpu_env
work/gpu_env/Scripts/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
work/gpu_env/Scripts/python -m pip install numpy==2.5.3 mmengine==0.10.7 mmcv-lite==2.2.0 opencv-python==5.0.0.93 einops==0.8.2 timm==1.0.29 scipy==1.18.1 pyyaml==6.0.3
```

- `pose_env`는 변경하지 않았다.
- `gpu_env`에는 rtmlib·onnxruntime·mediapipe를 설치하지 않았다. **좌표 추출(RTMPose)은 여전히 pose_env의 CPU 경로**다.

## 방법

- 입력: Pexels 8189172 (25FPS, 217프레임). 기존 RTMPose 좌표 `stgcn_raw/8189172/pose/observations.jsonl`와 RGB 프레임을 사용했다. 디코드·리사이즈는 `compare_pretrained.py`와 같다.
- 창: 10월 `timeline.json`의 100프레임 창 10개. 추가로 전체 영상을 공식 다중 뷰로 처리했다(ST-GCN++ 10클립, Swin 4클립×3crop).
- 모델 로더: `action_adapter.ActionModel`, `compare_pretrained.official_classes`를 그대로 썼다. 텐서를 장치로 옮기는 부분만 추가했다.
- 코드: `run_parity.py`(추론·저장), `compare_parity.py`(비교).
- 기준: pose_env CPU 출력.

## 결과 (확률 최대 절대차, top1/top5 순서 일치)

| 비교 | ST-GCN++ 창10 | ST-GCN++ 전체10클립 | Swin 창10 | Swin 전체12뷰 |
|---|---|---|---|---|
| 10월 저장 Swin CPU vs 오늘 pose_env CPU | – | – | **0.0**, 10/10 | – |
| pose_env CPU vs gpu_env CPU | 0.0, 10/10 | 0.0 | 0.0, 10/10 | 0.0 |
| pose_env CPU vs **gpu_env CUDA FP32** (TF32 끔) | 5.7e-7, top1 10/10, top5 10/10 | 2.4e-7 | 6.6e-7, 10/10, 10/10 | 3.5e-7 |
| pose_env CPU vs gpu_env CUDA TF32 | 5.1e-3, top1 10/10, **top5 9/10** | 1.2e-3 | 4.3e-4, 10/10, 10/10 | 1.8e-4 |

- 모든 창에서 top1과 2위 사이 점수 차의 최소값은 ST-GCN++ 0.044, Swin 0.061이었다. FP32 오차(1e-6 미만)보다 5자리 크다.
- **결론: TF32를 끈 CUDA FP32 출력은 기존 CPU 출력과 부동소수점 반올림 수준(1e-6 미만)에서 같다.** 이 입력에서는 top1과 top5 순서가 모두 같았다.
- TF32를 켜면 오차가 1e-3 수준으로 커진다. 그래서 ST-GCN++의 top5 순서가 한 창에서 바뀌었다(top1은 유지). **기존 결과와 비교하는 실험에서는 TF32를 끈다.**

## 처리 시간 (창 하나 모델 계산, 예열 뒤 중앙값; 디코드·좌표 추출 제외, 장치 간 복사 포함)

| | CPU 4스레드 | CUDA FP32 | CUDA TF32 |
|---|---|---|---|
| ST-GCN++ (100프레임, 1클립) | 0.024s | 0.007s | 0.009s |
| Swin Tiny (32장, 1뷰) | 1.24~1.30s | **0.039s** | 0.034s |

- Swin은 약 33배 빨라져서 0.48초 갱신 간격 안에 들어온다. 하지만 **좌표 추출·디코드를 포함한 전체 실시간 처리 속도는 측정하지 않았다.**
- TF32의 속도 이득은 작다(Swin 약 12%). ST-GCN++는 측정 잡음 범위다.

## 한계

- 영상 1개, 창 10개와 전체 영상 1회의 검사다. 다른 입력 분포에서도 같다는 것을 증명한 것은 아니다. 다만 같은 연산을 같은 가중치로 계산했으므로 차이는 수치 오차로 해석한다.
- 학습 경로(역전파, AMP, cudnn 비결정성)는 검증하지 않았다. 파인튜닝 전에 별도로 확인한다.
- 좌표 추출의 GPU 이전(onnxruntime-gpu 등)은 하지 않았다.

## 산출물

- `cpu_pose_env.npz`, `cpu_gpu_env.npz`, `cuda_gpu_env.npz`, `cuda_tf32_gpu_env.npz`: 확률과 메타데이터
- `report_*.json`: 비교 결과
