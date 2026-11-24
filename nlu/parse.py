"""텍스트 -> 작업 JSON 파이프라인.

parse_command(text, mode) -> (task, meta)
  mode = "rules"      : 규칙만
         "rules+llm"  : 규칙(안전장치) + Ollama LLM. 규칙이 block 이면 LLM 을 부르지 않는다.
                        LLM 이 실패하면 규칙 결과로 폴백(meta.parser = "rules(fallback)")
"""
from __future__ import annotations

import time

from nlu.rules import parse_rules
from nlu.validate import validate


def parse_command(text: str, mode: str = "rules+llm", scene_floors=(1, 2, 3), scene: str = "v2") -> tuple[dict, dict]:
    """scene="v3": 버튼 색 매핑 끔, LLM 프롬프트에 꼭대기층=max(scene_floors)·주행 안내."""
    t0 = time.perf_counter()
    v3 = scene == "v3"
    rule = parse_rules(text, use_color=not v3)
    meta = {"parser": "rules", "llm_ms": None, "rule": {k: rule[k] for k in ("intent", "floor", "floors", "block", "confident")},
            "llm": None, "llm_error": None}
    llm = None
    if mode == "rules+llm" and not rule["block"]:
        from nlu.llm_ollama import parse_llm
        llm, lm = parse_llm(text, top_floor=max(scene_floors), mobile=v3)
        meta["llm_ms"] = lm["llm_ms"]
        meta["llm_model"] = lm["model"]
        if llm is None:
            meta["parser"] = "rules(fallback)"
            meta["llm_error"] = lm["error"]
        else:
            meta["parser"] = "rules+llm"
            meta["llm"] = llm
    elif mode == "rules+llm":
        meta["parser"] = "rules(block)"
    task = validate(rule, llm, scene_floors)
    meta["parse_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return task, meta
