from pathlib import Path
import json
import collections

O = Path(__file__).resolve().parents[1] / "src/ppt_window_comparison"
rows = []
lines = [
    "PPT 영상 입력 길이 비교",
    "공통 종료 시점만 비교. 수치는 예측 빈도이며 정확도가 아님.",
    "전처리: RGB 32장, GCN 100장을 시간순으로 균등 선택(짧으면 인접 프레임 반복). 순환 반복 없음.",
    "기존 중앙 샘플 방식도 별도 대조. 같은 공간 전처리/가중치/종료 시점, 추가 학습 및 평활화 없음.",
    "25fps·960px 공통 변환을 거쳐 이전 PPT 실행의 정확한 재현은 아님.",
    "CPU 계산 지연은 재생 자막 시점에 반영하지 않음. 실시간 처리속도 검증과 별도.",
    "",
]
for f in O.glob("*/predictions.json"):
    r = json.loads(f.read_text(encoding="utf-8"))
    for key, us in r["conditions"].items():
        us = [u for u in us if u["end_s"] >= 3.96]
        for u in us:
            assert max(u["rgb_source_frames"]) <= round(u["end_s"] * 25)
            assert max(u["skeleton_source_frames"]) <= round(u["end_s"] * 25)
        a = dict(collections.Counter(u["rgb"]["top5"][0]["label_ko"] for u in us))
        b = dict(
            collections.Counter(
                u["skeleton"][0]["prediction"]["top5"][0]["label_ko"] for u in us if u["skeleton"]
            )
        )
        rows.append(dict(video=r["id"], condition=key, windows=len(us), rgb=a, skeleton=b))
        lines.append(f"{r['id']} / {key} / {len(us)}개 창\n RGB {a}\n GCN {b}")
(O / "matched_summary.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
(O / "결과요약.txt").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
