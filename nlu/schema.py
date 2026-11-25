"""명령 JSON 스키마 (팀 합의 고정).

{"intent": "press" | "unlock" | "wait" | "reject" | "clarify", "floor": int | null, "reason": str}
  press   : floor 층 버튼을 누른다 (scene_v2)
  unlock  : 우리 집 현관 도어락에 등록 비밀번호를 입력해 연다 (scene_v4). floor 는 null
  wait    : 대기·금지·정지 명령 -> 움직이지 않음
  reject  : 수행 불가(장면에 없는 층, 작업 범위 밖) -> 움직이지 않고 이유를 알림
  clarify : 모호함(층 없음, 여러 층) -> 되묻기. 움직이지 않음
"""
from __future__ import annotations

INTENTS = ("press", "unlock", "wait", "reject", "clarify")
MOVING_INTENTS = ("press", "unlock")

# Ollama structured output 용 JSON schema
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "floor": {"type": ["integer", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["intent", "floor", "reason"],
}


def make_task(intent: str, floor: int | None = None, reason: str = "") -> dict:
    assert intent in INTENTS, intent
    return {"intent": intent, "floor": floor, "reason": reason}


def is_valid(obj) -> bool:
    return (isinstance(obj, dict) and obj.get("intent") in INTENTS
            and (obj.get("floor") is None or isinstance(obj.get("floor"), int)))
