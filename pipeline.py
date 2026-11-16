"""한 번의 시도(episode) = 텍스트 -> 해석 -> 검증 -> 목표 위치 -> 실행 -> 판정 (+녹화).

run_demo.py 와 eval/run_eval.py 가 같은 함수를 쓴다 -> 실행부(runner)만 바꿔 같은 시드로 비교 가능.

runner
  ik_gt   : IK 상태 기계 + 시뮬레이터 좌표 (기준선)
  ik_cam  : IK 상태 기계 + 카메라 인식 (head_rgb RGB+depth)
  act     : ACT 정책 (policy/act_runner.py, LeRobot 체크포인트 필요)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from nlu.parse import parse_command
from sim.scene import FLOORS, make_scene

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs"))
RUNNERS = ("ik_gt", "ik_cam", "act")

_cache: dict = {}


def _locator(runner, sim):
    if runner == "ik_cam":
        from perception.camera import CameraLocator
        key = ("cam", id(sim.model))
        if key not in _cache:
            _cache[key] = CameraLocator(sim.model)
        return _cache[key]
    from perception.site_gt import locate_button
    return locate_button


def run_episode(command: str, seed: int, condition: str = "base", floor: int | None = None,
                expect: dict | None = None, parser: str = "rules+llm", runner: str = "ik_gt",
                video: str | Path | None = None, act_policy=None, task_override: dict | None = None,
                run_name: str | None = None, split: str | None = None) -> dict:
    """expect: 정답 {"intent", "floor"}. 없으면 해석 결과를 정답으로 간주(데모용).
    task_override: 해석을 건너뛰고 이 작업 JSON 으로 실행(되묻기 후 확정된 작업 등)."""
    sim = make_scene(seed, condition, floor=floor)
    t_wall0 = time.perf_counter()
    if task_override is not None:
        task, meta = task_override, {"parser": "given", "llm_ms": None}
    else:
        task, meta = parse_command(command, parser, scene_floors=FLOORS)
    t_parse = time.perf_counter() - t_wall0

    rec = None
    if video:
        from sim.recorder import Recorder
        rec = Recorder(sim, video, header={"seed": seed, "condition": f"{condition}/{sim.offset_kind}",
                                           "command": command, "parsed": {k: task[k] for k in ("intent", "floor")},
                                           "parser": meta["parser"], "runner": runner})
        sim.step_hooks.append(rec)
        rec.capture()
    status = rec.set_status if rec else None

    if runner == "act" and task["intent"] == "press":
        from policy.act_runner import execute_policy
        res = execute_policy(task, sim, act_policy, status=status)
    else:
        from control.press_fsm import execute
        res = execute(task, sim, locate=_locator(runner, sim), status=status)

    # ---- 최종 판정: '의도한' 작업 기준 ----
    exp = expect if expect is not None else {"intent": task["intent"], "floor": task.get("floor")}
    exp_floor = exp.get("floor") if exp.get("intent") == "press" else None
    j = sim.judge(exp_floor)
    parse_ok = (task["intent"] == exp["intent"] or (exp["intent"] != "press" and task["intent"] != "press")) \
        and (exp["intent"] != "press" or task.get("floor") == exp_floor)
    fail_type = None
    if not j["success"]:
        if not parse_ok:
            fail_type = "nlu"
        elif exp_floor is None:
            fail_type = "moved_when_forbidden"
        else:
            fail_type = res.get("fail_type") or ("wrong_button" if j["other_mm"] >= 3 else "contact")
    elif not parse_ok and exp["intent"] == "press":
        fail_type = "nlu"
    success = bool(j["success"] and parse_ok) if exp["intent"] == "press" else bool(j["success"])

    out = {
        "run": run_name or runner, "split": split, "seed": seed, "condition": condition,
        "offset_kind": sim.offset_kind, "panel_offset_m": [round(float(v), 4) for v in sim.panel_offset],
        "command": command, "expect": exp, "parsed": {k: task.get(k) for k in ("intent", "floor", "reason")},
        "parser": meta["parser"], "llm_ms": meta.get("llm_ms"), "llm_error": meta.get("llm_error"),
        "intent_match": task["intent"] == exp["intent"], "success": success, "peak_mm": j["peak_mm"], "other_mm": j["other_mm"],
        "peaks_mm": {str(k): round(v, 2) for k, v in j["peaks_mm"].items()},
        "fail_type": None if success else fail_type,
        # 작업 수행 시간 = 명령 해석(실제 벽시계) + 로봇 동작(시뮬레이션 시간)
        "t_task_s": round(t_parse + res["t_task"], 3), "t_parse_s": round(t_parse, 3), "t_motion_s": res["t_task"],
        "policy_ms": res.get("policy_ms"), "policy_ms_max": res.get("policy_ms_max"), "perception": res.get("perception"), "ik": res.get("ik"),
        "wall_s": round(time.perf_counter() - t_wall0, 2), "video": None,
    }
    if rec:
        if exp["intent"] == "press":
            label = "SUCCESS" if success else f"FAIL: {out['fail_type']}"
        else:
            label = {"wait": "대기 (정상)", "reject": "거절 (정상)", "clarify": "되묻기 (정상)"}.get(task["intent"], "")
            label = label if success else f"FAIL: {out['fail_type']}"
        rec.set_status(state="DONE", result={"ok": success, "label": label})
        rec.hold(1.5)
        sim.step_hooks.remove(rec)
        v = rec.close()
        out["video"] = (os.path.relpath(v, ROOT) if Path(v).resolve().is_relative_to(ROOT) else str(Path(v).resolve())) if v else None
    return out


def close_all():
    for v in _cache.values():
        if hasattr(v, "close"):
            v.close()
    _cache.clear()
