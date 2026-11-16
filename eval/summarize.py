"""시도 로그 -> 지표 (SR, 일반화 격차, 작업 시간 중앙값, 실패 유형, 명령 적합성, 지연) + 비교표.

사용법: python eval/summarize.py outputs/eval/*/trials.jsonl   -> 마크다운 비교표 출력
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def _sr(rows):
    return round(100.0 * sum(r["success"] for r in rows) / len(rows), 1) if rows else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def summarize(trials):
    base = [t for t in trials if t["split"] == "base"]
    var = [t for t in trials if t["split"] == "variation"]
    cmd = [t for t in trials if t["split"] == "commands"]
    both = base + var
    s = {
        "n_base": len(base), "n_variation": len(var), "n_commands": len(cmd),
        "sr_base": _sr(base), "sr_variation": _sr(var), "sr_total": _sr(both),
        "gen_gap_pp": round(_sr(base) - _sr(var), 1) if base and var else None,
        "t_task_median_s_success": _median([t["t_task_s"] for t in both if t["success"]]),
        "t_task_failed_s": [t["t_task_s"] for t in both if not t["success"]],
        "fail_types": dict(Counter(t["fail_type"] for t in both if not t["success"])),
        "fail_types_by_split": {k: dict(Counter(t["fail_type"] for t in v if not t["success"]))
                                for k, v in (("base", base), ("variation", var), ("commands", cmd))},
        "cmd_correct": sum(t["success"] for t in cmd), "cmd_intent_match": sum(t.get("intent_match", False) for t in cmd),
        "llm_ms_median": _median([t["llm_ms"] for t in trials if t.get("parser") == "rules+llm"]),
        "policy_ms_median": _median([t.get("policy_ms") for t in trials]),
        "perception_err_mm_median": _median([(t.get("perception") or {}).get("err_mm") for t in trials]),
        "parsers_used": dict(Counter(t["parser"] for t in trials)),
    }
    return s


def table(summaries):
    hdr = "| 실행부 | 기본 SR | 변화 SR | 전체 SR | 일반화 격차 | 성공 시 시간(중앙값) | 추론 지연 | 명령 적합성(5회) | 주요 실패 유형 |\n|---|---|---|---|---|---|---|---|---|"
    lines = [hdr]
    for s in summaries:
        lat = []
        if s.get("llm_ms_median"):
            lat.append(f"LLM {s['llm_ms_median']:.0f}ms")
        if s.get("policy_ms_median"):
            lat.append(f"정책 {s['policy_ms_median']:.1f}ms")
        ft = ", ".join(f"{k} {v}" for k, v in sorted(s["fail_types"].items(), key=lambda x: -x[1])) or "-"
        lines.append(f"| {s['run']} | {s['sr_base']}% | {s['sr_variation']}% | {s['sr_total']}% | {s['gen_gap_pp']}%p | "
                     f"{s['t_task_median_s_success']}s | {' / '.join(lat) or '해당 없음'} | {s['cmd_correct']}/{s['n_commands']} | {ft} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sums = []
    for p in sys.argv[1:]:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        s = summarize(rows)
        s["run"] = rows[0]["run"] if rows else Path(p).parent.name
        sums.append(s)
    print(table(sums))
