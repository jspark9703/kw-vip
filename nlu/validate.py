"""검증: 규칙(안전장치)과 LLM 결과를 합쳐 최종 작업 JSON 을 확정한다.

우선순위
 1. 규칙의 부정·대기 판정(block)은 항상 우선 -> wait
 2. 해석 결과의 층이 장면에 없으면 -> reject ("현재 장면에 없는 N층")
 3. press 인데 층이 없으면 -> clarify ("몇 층을 눌러 드릴까요?")
 4. 규칙이 명시적 층수 1개를 찾았는데 LLM 이 다른 층을 말하면 -> clarify (충돌 시 움직이지 않음)
"""
from __future__ import annotations

from nlu.schema import make_task

CLARIFY_Q = "몇 층을 눌러 드릴까요?"


def validate(rule: dict, llm: dict | None, scene_floors=(1, 2, 3)) -> dict:
    if rule.get("block"):
        return make_task("wait", None, rule.get("reason", "부정·대기 표현"))

    cand = llm if llm is not None else rule
    intent, floor = cand.get("intent"), cand.get("floor")

    if llm is not None and intent == "press" and len(rule.get("floors", [])) == 1 \
            and floor is not None and floor != rule["floors"][0]:
        return make_task("clarify", None, f"해석 충돌(규칙 {rule['floors'][0]}층 / LLM {floor}층). {CLARIFY_Q}")
    if llm is not None and len(rule.get("floors", [])) > 1 and intent == "press":
        return make_task("clarify", None, f"여러 층이 언급됨. {CLARIFY_Q}")

    if intent == "press":
        if floor is None:
            return make_task("clarify", None, CLARIFY_Q)
        if floor not in scene_floors:
            name = f"지하 {-floor}층" if floor < 0 else f"{floor}층"
            return make_task("reject", None, f"현재 장면에 없는 {name} 버튼입니다 (가능: {', '.join(map(str, scene_floors))}층)")
        return make_task("press", int(floor), cand.get("reason", ""))
    if intent == "clarify":
        return make_task("clarify", None, cand.get("reason") or CLARIFY_Q)
    if intent == "wait":
        return make_task("wait", None, cand.get("reason", "대기"))
    return make_task("reject", None, cand.get("reason", "수행할 수 없는 명령"))
