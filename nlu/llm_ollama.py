"""Ollama 로컬 LLM 파서 (JSON 출력 모드 + few-shot).

환경 변수
  OLLAMA_HOST       기본 http://127.0.0.1:11434   (노트북에서도 로컬 Ollama -> 오프라인 동작)
  KWVIP_LLM_MODEL   기본 qwen2.5:3b              (1.5B~3B instruct 후보를 테스트셋으로 비교해 선택)
  KWVIP_LLM_TIMEOUT 기본 20 (초)

외부 서버/클라우드 API 를 쓰지 않는다. Ollama 가 없거나 응답이 깨지면 None 을 돌려주고
상위(parse.py)에서 규칙 결과로 폴백한다.
"""
from __future__ import annotations

import json
import os
import time

import requests

from nlu.schema import JSON_SCHEMA, is_valid

HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
MODEL = os.environ.get("KWVIP_LLM_MODEL", "qwen2.5:3b")
TIMEOUT = float(os.environ.get("KWVIP_LLM_TIMEOUT", "20"))

SYSTEM = """너는 휠체어 로봇팔의 명령 해석기다. 사용자의 한국어 문장을 JSON 하나로만 바꾼다.
로봇이 할 수 있는 일은 '엘리베이터 버튼 하나 누르기'뿐이다.
intent 규칙:
- press: 특정 층 버튼을 눌러 달라는 요청. floor 에 층 번호(정수).
- wait: 누르지 말라, 기다려라, 멈춰라 같은 금지·대기·정지 요청. floor 는 null.
- reject: 버튼 누르기와 관계없는 일(물건 가져오기, 문 열기 등). floor 는 null.
- clarify: 버튼을 누르라는데 몇 층인지 알 수 없거나 여러 층을 동시에 말함. floor 는 null.
'꼭대기/맨 위/가장 높은 층'은 이 장면에서 3층, '맨 아래/가장 낮은/로비/1층 출구'는 1층이다.
지하층은 음수(지하 1층 = -1)로 적는다. 장면에 있는지는 판단하지 말고 말한 층을 그대로 적는다.
reason 은 10자 이내 한국어."""

FEWSHOT = [
    ("2층 눌러줘", {"intent": "press", "floor": 2, "reason": "2층 요청"}),
    ("엘리베이터 삼층 좀 선택해줄래", {"intent": "press", "floor": 3, "reason": "3층 요청"}),
    ("맨 위층으로 가고 싶어", {"intent": "press", "floor": 3, "reason": "꼭대기층"}),
    ("아직 누르지 말고 기다려", {"intent": "wait", "floor": None, "reason": "대기 요청"}),
    ("물 한 잔 가져다줘", {"intent": "reject", "floor": None, "reason": "범위 밖"}),
    ("버튼 좀 눌러줘", {"intent": "clarify", "floor": None, "reason": "층 미지정"}),
    ("9층 눌러", {"intent": "press", "floor": 9, "reason": "9층 요청"}),
]


def _messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs


def available(timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=timeout)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return MODEL in names or f"{MODEL}:latest" in names
    except Exception:
        return False


def parse_llm(text: str, model: str | None = None) -> tuple[dict | None, dict]:
    """-> (task 또는 None, meta{llm_ms, model, raw, error})"""
    model = model or MODEL
    meta = {"model": model, "llm_ms": None, "raw": None, "error": None}
    body = {"model": model, "messages": _messages(text), "stream": False,
            "format": JSON_SCHEMA, "options": {"temperature": 0, "num_predict": 80}, "keep_alive": "30m"}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{HOST}/api/chat", json=body, timeout=TIMEOUT)
        meta["llm_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        r.raise_for_status()
        raw = r.json()["message"]["content"]
        meta["raw"] = raw
        obj = json.loads(raw)
        if isinstance(obj.get("floor"), float) and obj["floor"].is_integer():
            obj["floor"] = int(obj["floor"])
        if not is_valid(obj):
            meta["error"] = "invalid_schema"
            return None, meta
        return {"intent": obj["intent"], "floor": obj.get("floor"), "reason": str(obj.get("reason", ""))}, meta
    except Exception as e:  # 연결 실패, 타임아웃, JSON 오류
        meta["llm_ms"] = meta["llm_ms"] or round((time.perf_counter() - t0) * 1000, 1)
        meta["error"] = f"{type(e).__name__}: {e}"[:200]
        return None, meta
