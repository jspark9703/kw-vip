"""scene_v3 상위 상태 기계: 텍스트 명령 -> 주행 -> 패널 확인·도킹 -> 숫자 판독 -> 피드백 누르기 -> 판단.

  (press 가 아니면) IDLE: 휠체어·팔 모두 정지 상태 유지 (wait / reject / clarify)
  STOW_ARM      팔을 접은 자세 유지, 그리퍼 닫기
  NAV_TO_LOBBY  지도 사전정보(패널 대략 위치)로 엘리베이터 홀 대기 지점까지 주행 (라이다 우회)
  FIND_PANEL    head 카메라로 패널 확인 (못 찾으면 좌우로 돌며 탐색)
  DOCK          패널 기준 도킹 자세로 이동 (접근 전 재인식, 직선 접근, 자세 유지)
  OBSERVE / READ_DIGITS   손목 카메라 관찰 자세에서 숫자 판독 -> 요청 층이 없으면 누르지 않고 reject
  APPROACH -> PRESS_SERVO -> HOLD -> RETRACT -> VERIFY (실패 시 재인식 후 재시도)
  STOW_ARM -> DONE

runner
  ik_cam : 모든 목표를 센서로 (head 패널 인식, wrist 숫자 판독)  <- 최종 방식
  ik_gt  : 도킹 목표·버튼 위치를 시뮬레이터 좌표로 (기준선). 주행은 같은 라이다 주행
"""
from __future__ import annotations

import numpy as np

from control.observe import observe_panel
from control.press_feedback import press_with_feedback
from control.press_fsm import PressConfig, PressExecutor, _Timeout
from models.make_scene_v3 import PANEL_NOMINAL
from nav.dock import dock, find_and_dock
from nav.follower import Navigator, summary
from perception.site_gt import locate_button
from sim.world import dock_pose

SIM_TIMEOUT = 200.0   # 미션 전체 시뮬레이션 시간 한도 (s)

_cache: dict = {}


def _sensors(model):
    key = id(model)
    if key not in _cache:
        from perception.digits import WristCamera
        from perception.panel import PanelLocator
        _cache[key] = (PanelLocator(model), WristCamera(model))
    return _cache[key]


def close_all():
    for v in _cache.values():
        for c in v:
            c.close()
    _cache.clear()


def run_mission(task: dict, sim, runner: str = "ik_cam", status=None) -> dict:
    st0 = status or (lambda **kw: None)
    t0 = sim.time
    stages = []

    def st(**kw):   # 주행·팔 모듈의 상태 변경도 단계 기록에 남긴다
        if "state" in kw and (not stages or stages[-1]["stage"] != kw["state"]):
            stages.append({"stage": kw["state"], "t": round(sim.time - t0, 2)})
        st0(**kw)

    def stage(name, **info):
        st(state=name)
        if info:
            stages[-1].update(info)

    out = {"fail_type": None, "nav": None, "dock": None, "digits": None, "press": None, "stages": stages}

    def finish(floor, fail=None, **kw):
        j = sim.judge(floor)
        if fail is None and floor is not None and not j["success"]:
            fail = "nav_collision" if j["collisions"] else ("wrong_button" if j["other_mm"] >= 3 else "contact")
        if floor is None and not j["success"]:
            fail = "moved_when_forbidden"
        stage("DONE", success=bool(j["success"] and fail is None))
        return {**j, **out, **kw, "success": bool(j["success"] and fail is None), "fail_type": fail,
                "t_task": round(sim.time - t0, 3)}

    # ---- 움직이면 안 되는 명령 ----
    if task.get("intent") != "press" or task.get("floor") is None:
        stage("IDLE")
        sim.hold()
        sim.step(int(2.0 / sim.dt))
        return finish(None)

    floor = int(task["floor"])
    try:
        return _press_mission(sim, floor, runner, st, stage, out, finish, t0)
    except _Timeout:
        return finish(floor, "timeout")


def _press_mission(sim, floor, runner, st, stage, out, finish, t0):
    ploc, wcam = _sensors(sim.model) if runner == "ik_cam" else (None, _sensors(sim.model)[1])
    arm = PressExecutor(sim, PressConfig(timeout_s=SIM_TIMEOUT), status=st)
    q_stow = sim.arm_q()

    stage("STOW_ARM")
    arm.move_ctrl(q_stow, 0.3, grip=0.0)

    # ---- 주행 + 패널 확인 + 도킹 ----
    nav = Navigator(sim, status=st)
    gt_panel = sim.data.xpos[sim.model.body("elevator_panel").id].copy()
    if runner == "ik_cam":
        r = find_and_dock(nav, ploc, PANEL_NOMINAL)
        det = r.get("panel_det")
    else:
        r = dock(nav, dock_pose(gt_panel))
        det = {"center": gt_panel, "normal": np.array([0.0, 1.0, 0.0])}
    gt_dk = dock_pose(gt_panel)
    out["nav"] = summary(nav)
    out["dock"] = {k: v for k, v in r.items() if k not in ("panel_det", "nav", "lobby")}
    out["dock"]["dock_err_gt_mm"] = round(float(np.linalg.norm(sim.base_pose()[:2] - gt_dk[:2]) * 1000), 1)
    if not r["ok"]:
        why = r["why"] or "dock"
        fail = "panel_not_found" if why == "panel_not_found" else ("nav" if why.startswith("nav") else "dock")
        return finish(floor, fail)

    # ---- 숫자 판독 ----
    def observe():
        ob = observe_panel(sim, wcam, det["center"], det["normal"], ex=arm)
        if ob["ok"] and runner != "ik_cam":   # 기준선: 버튼 위치를 시뮬레이터 좌표로 덮어씀
            ob["read"]["floors"] = {f: {**locate_button(sim, f), "score": 1.0} for f in sim.floors}
        return ob

    stage("OBSERVE")
    ob = observe()
    if not ob["ok"]:
        return finish(floor, "ik")
    read = ob["read"]
    # 판독 정확도(평가용): 읽은 층의 위치가 실제 그 층 버튼과 2cm 안이면 정답
    correct = sum(int(np.linalg.norm(np.asarray(v["face"]) - locate_button(sim, f)["face"]) < 0.02)
                  for f, v in read["floors"].items() if f in sim.floors)
    out["digits"] = {"read": sorted(int(f) for f in read["floors"]), "n_buttons": read.get("n_buttons"),
                     "correct": int(correct), "n_panel": len(sim.floors),
                     "scores": {int(f): v["score"] for f, v in read["floors"].items()}}
    if floor not in read["floors"]:
        # 패널에서 요청한 층 버튼을 찾지 못함 -> 누르지 않고 알림
        stage("REJECT_NOT_ON_PANEL")
        arm.move_ctrl(q_stow, 2.0)
        return finish(floor, "digit", message=f"패널에서 {floor}층 버튼을 찾지 못했습니다 (판독: {out['digits']['read']})")

    # ---- 피드백 누르기 + 판단 ----
    first = {"n": 0}

    def observe_cached():
        first["n"] += 1
        return ob if first["n"] == 1 else observe()

    pr = press_with_feedback(sim, floor, observe_cached, wcam, status=st)
    out["press"] = {k: pr[k] for k in ("n_attempts", "verify", "touch_trace")} | {"results": [a["result"] for a in pr["attempts"]]}
    stage("STOW_ARM")
    arm.move_ctrl(q_stow, 2.0)
    sim.step(int(0.3 / sim.dt))
    return finish(floor, pr["fail_type"])
