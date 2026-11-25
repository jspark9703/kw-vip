"""규칙 기반 파서.  역할 두 가지:
 1) 안전장치: 부정·대기·정지 표현을 먼저 걸러낸다 (LLM 결과보다 항상 우선).
 2) 단독 파서: 층수(숫자·한글 숫자·서수·버튼 색)와 누르기 동사를 뽑는다.
    버튼 색 -> 층 매핑은 scene_v2(색 버튼) 전용. (scene_v4 도어락 명령은 nlu/doorlock.py)

parse_rules(text) -> {"intent", "floor", "reason", "floors": [...], "block": bool, "confident": bool}
  block=True     : 부정·대기 판정 (validate 가 무조건 wait 로 확정)
  confident=True : 규칙만으로 확실히 해석했다고 보는 경우
"""
from __future__ import annotations

import re

from nlu.schema import make_task

# ---- 부정·대기·정지 (안전장치) ----
NEG_PATTERNS = [
    r"지\s*(마|말|않)",            # 누르지 마 / 하지 말고 / 누르지 않
    r"(안|못)\s*눌",                # 안 눌러도 돼
    r"누르면\s*안",
    r"하면\s*안",
    r"필요\s*(없|없어)",
    r"기다려|기다리|대기",
    r"멈춰|멈추|정지|스톱|그만|취소|가만히|가만\s*있",
    r"\bstop\b|\bwait\b|don'?t|do not|\bhold\b|\bcancel\b",
]
# "2층 말고 3층" 처럼 '층 + 말고' 는 부정이 아니라 대상 교체
FLOOR_MALGO = re.compile(r"(\d+|[일이삼사오육칠팔구십]+)\s*층\s*(은|는)?\s*말고")

# ---- 누르기 동사 / 작업 단서 ----
PRESS_PATTERNS = [
    r"눌러|누르|눌|누름", r"선택", r"터치|클릭|찍어|입력", r"가\s*(줘|주|자|고\s*싶)|갈래|갑시다|가야",
    r"올라|내려", r"부탁", r"\bpress\b|\bpush\b|\bhit\b|\bselect\b|go to",
]
CONTEXT_PATTERNS = [r"버튼", r"엘리베이터", r"층", r"\bfloor\b|\bbutton\b|elevator"]

KO_NUM = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9, "십": 10}
ORDINAL = {"첫": 1, "첫째": 1, "첫번째": 1, "두번째": 2, "둘째": 2, "세번째": 3, "셋째": 3,
           "네번째": 4, "넷째": 4, "다섯번째": 5}
COLOR = {1: r"빨간|빨강|붉은|red", 2: r"초록|녹색|green", 3: r"파란|파랑|blue"}


def _ko_number(tok: str) -> int | None:
    # 일, 이, 삼 ... 십, 십일, 이십 정도까지
    if not tok or any(c not in KO_NUM for c in tok):
        return None
    if "십" in tok:
        a, _, b = tok.partition("십")
        return (KO_NUM.get(a, 1) if a else 1) * 10 + (KO_NUM.get(b, 0) if b else 0)
    return KO_NUM[tok] if len(tok) == 1 else None


def extract_floors(text: str, use_color: bool = True) -> tuple[list[int], list[int]]:
    """(언급된 층 목록, '말고'로 제외된 층 목록). 순서 유지, 중복 제거."""
    t = text.lower()
    found: list[tuple[int, int]] = []  # (위치, 층)
    excluded: list[int] = []
    for mm in FLOOR_MALGO.finditer(t):
        v = int(mm.group(1)) if mm.group(1).isdigit() else _ko_number(mm.group(1))
        if v is not None:
            excluded.append(v)
    # 지하 n층 -> 음수 (장면에 없음)
    for mm in re.finditer(r"(지하|b)\s*(\d+)\s*(층|f)?", t):
        found.append((mm.start(), -int(mm.group(2))))
    t_wo_b = re.sub(r"(지하|b)\s*\d+\s*(층|f)?", " ", t)
    for mm in re.finditer(r"(\d+)\s*(층|f\b|번\s*층|floor)", t_wo_b):
        found.append((mm.start(), int(mm.group(1))))
    for mm in re.finditer(r"floor\s*(\d+)|(\d+)(st|nd|rd|th)\s*floor", t_wo_b):
        found.append((mm.start(), int(mm.group(1) or mm.group(2))))
    # 한글 숫자 + 층 (띄어쓰기 없는 경우만: '이 층'은 '이(this) 층'일 수 있어 제외)
    for mm in re.finditer(r"(?<![가-힣])([일이삼사오육칠팔구십]{1,3})층", t_wo_b):
        v = _ko_number(mm.group(1))
        if v is not None:
            found.append((mm.start(), v))
    # 서수 + (번째) 층/버튼
    for mm in re.finditer(r"(첫\s*번?째|첫|두\s*번째|둘째|세\s*번째|셋째|네\s*번째|넷째|다섯\s*번째)\s*(층|버튼)", t):
        key = re.sub(r"\s", "", mm.group(1))
        if key in ORDINAL:
            found.append((mm.start(), ORDINAL[key]))
    # 버튼 색 (scene_v2: 1층 빨강, 2층 초록, 3층 파랑)
    for fl, pat in (COLOR.items() if use_color else ()):
        for mm in re.finditer(pat, t):
            found.append((mm.start(), fl))
    out: list[int] = []
    for _, v in sorted(found):
        if v not in out:
            out.append(v)
    return [v for v in out if v not in excluded], excluded


def has_any(patterns, text) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def is_negated(text: str) -> bool:
    t = text.lower()
    # "2층 말고 3층" 형태는 지우고 판정
    t = FLOOR_MALGO.sub(" ", t)
    return has_any(NEG_PATTERNS, t)


def parse_rules(text: str, use_color: bool = True) -> dict:
    floors, excluded = extract_floors(text, use_color)
    press = has_any(PRESS_PATTERNS, text)
    context = has_any(CONTEXT_PATTERNS, text) or bool(floors)
    res = {"floors": floors, "excluded": excluded, "press_verb": press, "block": False, "confident": False}
    if is_negated(text):
        return {**res, **make_task("wait", None, "부정·대기 표현"), "block": True, "confident": True}
    if len(floors) > 1:
        return {**res, **make_task("clarify", None, "층이 여러 개 언급됨"), "confident": True}
    if len(floors) == 1:
        f = floors[0]
        if press or re.search(r"층\s*(이요|요|으로|에|까지)?\s*[.!?]*\s*$", text.strip()):
            return {**res, **make_task("press", f, "층수+누르기 표현"), "confident": press}
        return {**res, **make_task("press", f, "층수만 있음"), "confident": False}
    if press and context:
        return {**res, **make_task("clarify", None, "층수가 없음")}
    return {**res, **make_task("reject", None, "버튼 누르기와 관련 없는 명령")}
