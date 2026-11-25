"""scene_v4 상위 상태 기계: 텍스트 명령 -> 주행 -> 도어락 확인·도킹 -> 키 글자 판독 -> 비밀번호 입력(피드백) -> 판단.

  (unlock 이 아니면) IDLE: 휠체어·팔 모두 정지 상태 유지 (wait / reject / clarify)
  STOW_ARM      팔을 접은 자세 유지, 그리퍼 닫기
  NAV_TO_LOBBY  지도 사전정보(우리 집 도어락 대략 위치)로 현관 앞 대기 지점까지 주행 (라이다로 택배·사람 우회)
  FIND_PANEL    head 카메라로 도어락 확인 (못 찾으면 좌우로 돌며 탐색, 이웃집 도어락은 사전정보 거리로 배제)
  DOCK          도어락 기준 도킹 자세로 이동 (접근 전 재인식, 직선 접근, 자세 유지)
  OBSERVE / READ_KEYS   손목 카메라 관찰 자세에서 12개 키 글자 판독 -> 비밀번호에 필요한 키가 없으면 누르지 않고 알림
  APPROACH_k -> PRESS_k (키마다 피드백) ... '#' -> VERIFY (상태 LED 초록 + 키 센서 입력 기록)
  STOW_ARM -> DONE

비밀번호: 로봇 설정에 미리 등록된 값(sim.password = 이 에피소드의 등록 비밀번호). 명령 문장에는 숫자가 없다.

runner
  ik_cam : 모든 목표를 센서로 (head 도어락 인식, wrist 키 판독)  <- 최종 방식
  ik_gt  : 도킹 목표·키 위치를 시뮬레이터 좌표로 (기준선). 주행은 같은 라이다 주행, LED 확인은 카메라
"""
from __future__ import annotations

import numpy as np

from control.observe import observe_panel
from control.press_feedback import enter_password
from control.press_fsm import PressConfig, PressExecutor, _Timeout
from models.make_scene_v4 import DOOR_NOMINAL_X, LOCK_REL_NOMINAL
from nav.dock import dock, find_and_dock
from nav.follower import Navigator, summary
from perception.site_gt import locate_key
from sim.world import dock_pose

SIM_TIMEOUT = 240.0   # 미션 전체 시뮬레이션 시간 한도 (s)
PRIOR_LOCK = np.array([DOOR_NOMINAL_X + LOCK_REL_NOMINAL[0], LOCK_REL_NOMINAL[1], LOCK_REL_NOMINAL[2]])  # 지도 사전정보

_cache: dict = {}


def _sensors(model):
    key = id(model)
    if key not in _cache:
        from perception.digits import WristCamera
        from perception.panel import PanelLocator
        _cache[key] = (PanelLocator(model, prior=PRIOR_LOCK), WristCamera(model))
    return _cache[key]


def close_all():
    for v in _cache.values():
        for c in v:
            c.close()
    _cache.clear()


def run_mission(task: dict, sim, runner: str = "ik_cam", status=None, password: str | None = None) -> dict:
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

    out = {"fail_type": None, "nav": None, "dock": None, "keys": None, "press": None, "stages": stages}

    def finish(pw, fail=None, **kw):
        j = sim.judge(pw)
        if fail is None and pw is not None and not j["success"]:
            fail = "nav_collision" if j["collisions"] else ("wrong_key" if j.get("wrong_inputs") else "unlock_failed")
        if pw is None and not j["success"]:
            fail = "moved_when_forbidden"
        stage("DONE", success=bool(j["success"] and fail is None))
        return {**j, **out, **kw, "success": bool(j["success"] and fail is None), "fail_type": fail,
                "t_task": round(sim.time - t0, 3)}

    # ---- 움직이면 안 되는 명령 ----
    if task.get("intent") != "unlock":
        stage("IDLE")
        sim.hold()
        sim.step(int(2.0 / sim.dt))
        return finish(None)

    pw = password or sim.password          # 로봇 설정의 등록 비밀번호
    try:
        return _unlock_mission(sim, pw, runner, st, stage, out, finish)
    except _Timeout:
        return finish(pw, "timeout")


def _unlock_mission(sim, pw, runner, st, stage, out, finish):
    ploc, wcam = _sensors(sim.model)
    arm = PressExecutor(sim, PressConfig(timeout_s=SIM_TIMEOUT), status=st)
    q_stow = sim.arm_q()

    stage("STOW_ARM")
    arm.move_ctrl(q_stow, 0.3, grip=0.0)

    # ---- 주행 + 도어락 확인 + 도킹 ----
    nav = Navigator(sim, status=st)
    gt_lock = sim.lock_pos()
    if runner == "ik_cam":
        r = find_and_dock(nav, ploc, PRIOR_LOCK)
        det = r.get("panel_det")
    else:
        r = dock(nav, dock_pose(gt_lock))
        det = {"center": gt_lock, "normal": np.array([0.0, 1.0, 0.0])}
    gt_dk = dock_pose(gt_lock)
    out["nav"] = summary(nav)
    out["dock"] = {k: v for k, v in r.items() if k not in ("panel_det", "nav", "lobby")}
    out["dock"]["dock_err_gt_mm"] = round(float(np.linalg.norm(sim.base_pose()[:2] - gt_dk[:2]) * 1000), 1)
    if not r["ok"]:
        why = r["why"] or "dock"
        fail = "panel_not_found" if why == "panel_not_found" else ("nav" if why.startswith("nav") else "dock")
        return finish(pw, fail)

    # ---- 키 글자 판독 ----
    def observe(aim=0.0):
        ob = observe_panel(sim, wcam, det["center"], det["normal"], ex=arm, aim=aim)
        if ob["ok"] and runner != "ik_cam":   # 기준선: 키 위치를 시뮬레이터 좌표로 덮어씀
            ob["read"]["keys"] = {k: {**locate_key(sim, k), "score": 1.0} for k in sim.keys}
        return ob

    stage("OBSERVE")
    ob = observe()
    if not ob["ok"]:
        return finish(pw, "ik")
    read = ob["read"]
    # 판독 정확도(평가용): 읽은 키 위치가 실제 그 키와 8mm 안이면 정답
    correct = sum(int(np.linalg.norm(np.asarray(v["face"]) - locate_key(sim, k)["face"]) < 0.008)
                  for k, v in read["keys"].items())
    out["keys"] = {"read": "".join(sorted(read["keys"])), "n_found": read.get("n_keys"), "correct": int(correct),
                   "n_keys": len(sim.keys), "min_score": round(min([v["score"] for v in read["keys"].values()] or [0]), 3)}
    missing = sorted(set(pw + "#") - set(read["keys"]))
    if missing:
        stage("REJECT_KEYS_NOT_FOUND")
        arm.move_ctrl(q_stow, 2.0)
        return finish(pw, "digit", message=f"도어락에서 {', '.join(missing)} 키를 찾지 못해 입력하지 않았습니다")

    # ---- 비밀번호 입력 (키마다 피드백) + 판단 ----
    pr = enter_password(sim, pw, observe, wcam, status=st, first_read=ob)
    out["press"] = {k: pr[k] for k in ("n_presses", "restarts", "verify", "entered", "trace")} | \
        {"results": [p["result"] for p in pr["presses"]]}
    stage("STOW_ARM")
    arm.move_ctrl(q_stow, 2.0)
    sim.step(int(0.3 / sim.dt))
    return finish(pw, pr["fail_type"])
