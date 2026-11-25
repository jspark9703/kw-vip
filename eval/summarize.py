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
    for sp in sorted({t["split"] for t in trials} - {"base", "variation", "commands"}):   # 추가 split (반복·한계)
        rows = [t for t in trials if t["split"] == sp]
        s[f"sr_{sp}"] = _sr(rows)
        s[f"n_{sp}"] = len(rows)
        s[f"fail_types_{sp}"] = dict(Counter(t["fail_type"] for t in rows if not t["success"]))
        if rows and rows[0].get("scene") == "v4":
            s[f"subtasks_{sp}"] = subtasks(rows)
    v4 = [t for t in both if t.get("scene") == "v4"]
    if v4:
        s["subtasks"] = subtasks(v4)
    return s


def subtasks(rows):
    """scene_v4 서브태스크별 지표 (주행·도킹·키 판독·비밀번호 입력·판단)."""
    n = len(rows)
    dock_ok = [t for t in rows if (t.get("dock") or {}).get("ok")]
    keys = [t["keys"] for t in rows if t.get("keys")]
    pr = [t["press"] for t in rows if t.get("press")]
    ver = [p["verify"] for p in pr if p.get("verify")]

    def stage_t(t, name):
        return next((tt for st, tt in t.get("stages", []) if st == name), None)
    return {
        "n": n,
        "nav_dock_reached": f"{len(dock_ok)}/{n}",
        "collision_trials": sum(1 for t in rows if (t.get("collisions") or 0) > 0),
        "replans_median": _median([(t.get("nav") or {}).get("replans") for t in rows]),
        "estops_total": sum((t.get("nav") or {}).get("estops") or 0 for t in rows),
        "travel_m_median": _median([(t.get("nav") or {}).get("traveled_m") for t in rows]),
        "panel_err_mm_median": _median([((t.get("dock") or {}).get("panel_refined") or (t.get("dock") or {}).get("panel_first") or {}).get("err_mm") for t in rows]),
        "dock_err_mm_median": _median([(t.get("dock") or {}).get("dock_err_gt_mm") for t in dock_ok]),
        "back_side_approach": sum(1 for t in dock_ok if t["dock"].get("side") == "back"),
        "key_read_correct": f"{sum(k.get('correct', 0) for k in keys)}/{sum(k.get('n_keys', 12) for k in keys)}",
        "keys_pressed": sum(len(t.get("target") or "") for t in rows if t.get("unlocked")),
        "presses_per_key": round(sum(p["n_presses"] for p in pr) / max(1, sum(len(t.get("target") or "") for t in rows if t.get("press"))), 3) if pr else None,
        "restart_trials": sum(1 for p in pr if p.get("restarts")),
        "wrong_inputs": sum(t.get("wrong_inputs") or 0 for t in rows),
        "led_sensor_agree": f"{sum(bool(v.get('agree')) for v in ver)}/{len(ver)}",
        "t_reach_panel_s_median": _median([stage_t(t, "OBSERVE") for t in rows]),
        "t_entry_s_median": _median([(t["t_motion_s"] - stage_t(t, "OBSERVE")) if stage_t(t, "OBSERVE") is not None else None for t in rows]),
    }


def subtask_table(summaries):
    keys = [("nav_dock_reached", "도킹 도달"), ("collision_trials", "충돌 시도"), ("estops_total", "비상정지"),
            ("travel_m_median", "주행 거리"), ("panel_err_mm_median", "패널 오차mm"), ("dock_err_mm_median", "도킹 오차mm"),
            ("back_side_approach", "반대쪽 접근"), ("key_read_correct", "키 판독"), ("keys_pressed", "누른 키 수"),
            ("presses_per_key", "키당 누름"), ("restart_trials", "* 재입력"), ("wrong_inputs", "틀린 입력"),
            ("led_sensor_agree", "LED=센서"), ("t_reach_panel_s_median", "도어락 도착s"), ("t_entry_s_median", "입력 단계s")]
    rows = [s for s in summaries if s.get("subtasks")]
    if not rows:
        return ""
    out = ["| 실행부 | " + " | ".join(k[1] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for s in rows:
        out.append(f"| {s['run']} | " + " | ".join(str(s["subtasks"].get(k[0])) for k in keys) + " |")
    return "\n".join(out)


def table(summaries):
    hdr = "| 실행부 | 기본 SR | 변화 SR | 전체 SR | 일반화 격차 | 성공 시 시간(중앙값) | 추론 지연 | 명령 적합성 | 주요 실패 유형 |\n|---|---|---|---|---|---|---|---|---|"
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
    st = subtask_table(sums)
    if st:
        print("\n서브태스크 (scene v4, 기본+변화 시도)\n" + st)
    for s in sums:
        for k in sorted(k for k in s if k.startswith("sr_") and k not in ("sr_base", "sr_variation", "sr_total")):
            sp = k[3:]
            print(f"\n[{s['run']}] 추가 split '{sp}': SR {s[k]}% (n={s[f'n_{sp}']}), 실패 {s.get(f'fail_types_{sp}')}")
            if s.get(f"subtasks_{sp}"):
                print(subtask_table([{"run": f"{s['run']}/{sp}", "subtasks": s[f"subtasks_{sp}"]}]))
