"""scene_v4 명령 해석: 우리 집 현관 도어락 열기 (등록 비밀번호 입력).

규칙(안전장치 + 단독 파서) -> parse_rules_v4(text)
  1) 부정·대기·정지 표현(nlu/rules.is_negated)      -> wait (block: LLM 결과보다 항상 우선)
  2) 다른 집(옆집, 이웃, 우리 호수가 아닌 'N호')    -> reject
  3) 명령에 비밀번호 숫자를 직접 말함               -> reject (등록된 비밀번호만 입력한다는 설계 결정)
  4) 문 열기·현관·도어락·비밀번호·집에 들어가기       -> unlock
  5) 누르기 동사만 있고 무엇을 할지 모름 / 색으로 지시 -> clarify ("현관 도어락 비밀번호를 입력할까요?")
  6) 그 밖                                           -> reject (작업 범위 밖)
validate_v4(rule, llm): block 은 항상 wait, 규칙의 reject(다른 집·숫자 비밀번호)도 LLM 보다 우선,
  규칙 unlock 인데 LLM 이 다른 것을 말하면 clarify (충돌 시 움직이지 않음).
"""
from __future__ import annotations

import re

from nlu.rules import has_any, is_negated
from nlu.schema import make_task

HOME_UNIT = "1203"
CLARIFY_Q = "현관 도어락에 등록된 비밀번호를 입력할까요?"

UNLOCK_PATTERNS = [
    r"문\s*(좀|을|를)?\s*(열|따)", r"현관", r"도어\s*락", r"비밀\s*번호", r"비번", r"잠금\s*(을|를)?\s*(풀|해제)",
    r"집\s*(에|으로|안으로)?\s*(들어|도착)", r"들어가(자|고\s*싶|야|게)", r"\bunlock\b", r"open\s+the\s+door", r"\bdoor\b",
]
PRESS_PATTERNS = [r"눌러|누르|눌|누름|입력|터치|찍어", r"\bpress\b|\bpush\b|\benter\b"]
OTHER_HOME = [r"옆\s*집", r"이웃", r"남의\s*집", r"다른\s*(집|호|세대)", r"윗\s*집", r"아랫\s*집"]
COLOR = r"빨간|빨강|초록|녹색|파란|파랑|노란|검은|흰|red|green|blue"


def parse_rules_v4(text: str) -> dict:
    t = text.lower()
    units = re.findall(r"(\d{3,4})\s*호", t)
    digits = re.findall(r"\d", re.sub(r"\d{3,4}\s*호", " ", t))
    res = {"floors": [], "excluded": [], "block": False, "confident": False, "units": units}
    if is_negated(text):
        return {**res, **make_task("wait", None, "부정·대기 표현"), "block": True, "confident": True}
    if has_any(OTHER_HOME, t) or any(u != HOME_UNIT for u in units):
        return {**res, **make_task("reject", None, f"우리 집({HOME_UNIT}호)이 아닌 집의 문은 열지 않습니다"), "confident": True}
    unlock = has_any(UNLOCK_PATTERNS, t)
    if len(digits) >= 3:
        return {**res, **make_task("reject", None, "비밀번호는 등록된 값만 입력합니다 (말한 숫자는 입력하지 않음)"),
                "confident": True}
    if unlock:
        return {**res, **make_task("unlock", None, "현관문 열기"), "confident": True}
    if has_any(PRESS_PATTERNS, t) or re.search(COLOR, t):
        return {**res, **make_task("clarify", None, CLARIFY_Q)}
    return {**res, **make_task("reject", None, "현관 도어락 열기와 관련 없는 명령")}


def validate_v4(rule: dict, llm: dict | None) -> dict:
    if rule.get("block"):
        return make_task("wait", None, rule.get("reason", "부정·대기 표현"))
    if rule.get("intent") == "reject" and rule.get("confident"):
        return make_task("reject", None, rule["reason"])
    cand = llm if llm is not None else rule
    intent = cand.get("intent")
    if llm is not None and rule.get("intent") == "unlock" and intent != "unlock":
        return make_task("clarify", None, f"해석 충돌(규칙 unlock / LLM {intent}). {CLARIFY_Q}")
    if intent == "unlock":
        return make_task("unlock", None, cand.get("reason") or "현관문 열기")
    if intent == "press":                  # LLM 이 v2 습관으로 press 를 내면 되묻기
        return make_task("clarify", None, CLARIFY_Q)
    if intent == "clarify":
        return make_task("clarify", None, cand.get("reason") or CLARIFY_Q)
    if intent == "wait":
        return make_task("wait", None, cand.get("reason", "대기"))
    return make_task("reject", None, cand.get("reason", "수행할 수 없는 명령"))


SYSTEM_V4 = f"""너는 전동 휠체어 로봇팔의 명령 해석기다. 사용자의 한국어 문장을 JSON 하나로만 바꾼다.
로봇이 할 수 있는 일은 '아파트 복도를 지나 우리 집({HOME_UNIT}호) 현관 도어락에 미리 등록된 비밀번호를 눌러 문을 여는 것'뿐이다.
intent 규칙:
- unlock: 우리 집 현관문을 열어 달라, 집에 들어가자, 도어락 비밀번호를 눌러 달라는 요청. floor 는 null.
- wait: 누르지 말라, 기다려라, 멈춰라, 열지 말라 같은 금지·대기·정지 요청.
- reject: 다른 집 문, 비밀번호 숫자를 직접 불러 주는 요청, 문 열기와 관계없는 일(물 가져오기, 엘리베이터 등).
- clarify: 무언가 누르라는데 무엇을 할지 알 수 없을 때.
floor 는 항상 null 로 적는다."""
