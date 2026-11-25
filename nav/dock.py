"""도킹: 도킹 자세(x, y, yaw=0) 앞 PRE_DIST 지점까지 주행 -> yaw 정렬 -> 직선 접근.

앞쪽(-x)에서 접근이 막혀 있으면(장애물이 도킹 구역 뒤에 있는 경우) 반대쪽(+x)에서 후진으로 접근한다.
"""
from __future__ import annotations

import numpy as np

from nav.grid import ROBOT_R
from nav.planner import astar

PRE_DIST = 0.6
LAT_TOL = 0.01   # m, 도킹 횡오차 허용


def pre_dock_candidates(dock):
    u = np.array([np.cos(dock[2]), np.sin(dock[2])])
    return [("front", dock[:2] - PRE_DIST * u), ("back", dock[:2] + PRE_DIST * u)]


def choose_pre_dock(grid, pose, dock):
    """경로가 있는 후보 중 경로가 짧은 것. 없으면 기본(front)."""
    best = None
    for name, p in pre_dock_candidates(dock):
        if grid.clearance_at(p) < ROBOT_R or not grid.segment_free(p, dock[:2], ROBOT_R * 0.9):
            continue
        path = astar(grid, pose[:2], p, ROBOT_R)
        if path is None:
            continue
        L = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
        if best is None or L < best[2]:
            best = (name, p, L)
    return (best[0], best[1]) if best else pre_dock_candidates(dock)[0]


def dock(nav, dock_pose, refine=None) -> dict:
    """refine(nav) -> 새 dock_pose 또는 None : 접근 직전 다시 인식해서 목표 보정 (Step 4 카메라)."""
    dock_pose = np.asarray(dock_pose, float)
    side, pre = choose_pre_dock(nav.grid, nav.sim.base_pose(), dock_pose)
    cands = dict(pre_dock_candidates(dock_pose))
    other = [k for k in cands if k != side][0]
    r = nav.go_to(pre, tol=0.05, alts=[cands[other]])     # 가는 도중 장애물이 보여 막히면 반대쪽 접근으로 바꿈
    side = side if r.get("goal_idx", 0) == 0 else other
    out = {"side": side, "nav": r}
    if not r["ok"]:
        return {**out, "ok": False, "why": r["why"]}
    nav.status(state="ALIGN")
    r = nav.turn_to(dock_pose[2])
    if not r["ok"]:
        return {**out, "ok": False, "why": "dock_align"}
    if refine is not None:
        new = refine(nav)
        if new is not None:
            out["refined_shift_m"] = float(np.linalg.norm(np.asarray(new)[:2] - dock_pose[:2]))
            dock_pose = np.asarray(new, float)
            nav.turn_to(dock_pose[2])
    nav.status(state="DOCK")
    u = np.array([np.cos(dock_pose[2]), np.sin(dock_pose[2])]) * (1 if side == "front" else -1)
    for attempt in range(3):
        r = nav.approach_line(dock_pose[:2], dock_pose[2])
        if not r["ok"]:
            return {**out, "ok": False, "why": r["why"]}
        if abs(r["lat_err"]) <= LAT_TOL:
            break
        # 횡오차가 크면 0.4m 물러났다가 다시 직선 접근 (차동구동은 옆으로 못 가므로)
        nav.approach_line(dock_pose[:2] - 0.4 * u, dock_pose[2], v=0.2, tol=0.02)
    out["dock_attempts"] = attempt + 1
    nav.turn_to(dock_pose[2], tol=np.radians(0.5))
    nav.sim.hold(dock_pose)
    nav.sim.step(int(0.3 / nav.sim.dt))
    pose = nav.sim.base_pose()
    out.update({"ok": True, "why": None, "dock_target": dock_pose.tolist(),
                "dock_err_mm": float(np.linalg.norm(pose[:2] - dock_pose[:2]) * 1000),
                "dock_yaw_err_deg": float(np.degrees(abs((pose[2] - dock_pose[2] + np.pi) % (2 * np.pi) - np.pi)))})
    return out


# ---------------- 패널 찾기 + 도킹 (Step 4: 카메라) ----------------
HEAD_YAW_OFFSET = np.radians(-53.13)   # head_rgb 시선 = 휠체어 진행 방향에서 오른쪽 53° (xyaxes 로부터)
SCAN_OFFSETS = np.radians([0, 25, -25, 50, -50])


def lobby_point(prior_panel):
    """지도 사전정보(도어락 대략 위치) -> 도어락을 바라볼 대기 지점: 명목 도킹 자세 1.0m 뒤, 1.1m 바깥(복도 가운데 쪽).
    벽 가까운 접근로는 택배·사람이 놓이기 쉬워(패널 쪽 장애물) 복도 가운데에서 비스듬히 본다."""
    dk = _dock_nominal(prior_panel)
    return dk[:2] + np.array([-1.0, 1.1])


def lobby_alternatives(prior_panel, lob, max_dist=2.6):
    """대기 지점이 장애물로 막혔을 때 쓸 후보들: 벽에서 0.9~2.3m, 패널에서 max_dist 이내, 선호 지점에서 가까운 순."""
    pts = []
    for dx in np.arange(-1.2, 1.21, 0.3):
        for y in np.arange(0.9, 2.31, 0.35):
            p = np.array([lob[0] + dx, y])
            if np.linalg.norm(p - lob) < 0.1 or np.linalg.norm(p - np.asarray(prior_panel)[:2]) > max_dist:
                continue
            pts.append(p)
    pts.sort(key=lambda p: np.linalg.norm(p - lob))
    return pts


def _dock_nominal(panel):
    from perception.panel import dock_from_panel
    return dock_from_panel(panel, [0.0, 1.0, 0.0])


def look_at_panel(nav, locate, target_xyz, offsets=SCAN_OFFSETS):
    """head 카메라가 target 쪽을 보도록 제자리 회전 후 인식. 못 찾으면 좌우로 돌며 다시."""
    pos = nav.sim.base_pose()[:2]
    bearing = np.arctan2(target_xyz[1] - pos[1], target_xyz[0] - pos[0])
    base_yaw = bearing - HEAD_YAW_OFFSET
    for k, off in enumerate(offsets):
        nav.turn_to(base_yaw + off, tol=np.radians(2.0))
        nav.stop(0.2)
        r = locate(nav.sim)
        if r is not None:
            return r, k + 1
    return None, len(offsets)


def find_and_dock(nav, locate, prior_panel) -> dict:
    """locate(sim) -> {"center","normal","dock",...} | None  (perception/panel.PanelLocator)."""
    prior_panel = np.asarray(prior_panel, float)
    nav.status(state="NAV_TO_LOBBY")
    lob = lobby_point(prior_panel)
    r = nav.go_to(lob, tol=0.15, alts=lobby_alternatives(prior_panel, lob))
    out = {"lobby": r}
    if not r["ok"]:
        return {**out, "ok": False, "why": r["why"], "stage": "NAV_TO_LOBBY"}
    nav.status(state="FIND_PANEL")
    det, n_views = look_at_panel(nav, locate, prior_panel)
    # 못 찾으면(사람·택배가 시야를 가림, 멀어서 작게 보임) 사전정보 위치에 더 가까운 다른 시점으로 옮겨 최대 3번 다시 본다
    tried = 0
    if det is None:
        here = nav.sim.base_pose()[:2]
        views = sorted((p for p in lobby_alternatives(prior_panel, lob, max_dist=2.0)
                        if np.linalg.norm(p - here) > 0.4 and nav.grid.clearance_at(p) >= ROBOT_R),
                       key=lambda p: np.linalg.norm(p - np.asarray(prior_panel)[:2]))
        for p in views[:3]:
            tried += 1
            if not nav.go_to(p, tol=0.15)["ok"]:
                continue
            det, k = look_at_panel(nav, locate, prior_panel)
            n_views += k
            if det is not None:
                break
    out["find_views"] = n_views
    out["find_moves"] = tried
    if det is None:
        return {**out, "ok": False, "why": "panel_not_found", "stage": "FIND_PANEL"}
    out["panel_first"] = {"err_mm": round(det["err_mm"], 1)}
    latest = {"det": det}

    def refine(nv):
        d2, _ = look_at_panel(nv, locate, det["center"], offsets=np.radians([0, 15, -15]))
        if d2 is None:
            return None
        out["panel_refined"] = {"err_mm": round(d2["err_mm"], 1)}
        latest["det"] = d2
        return d2["dock"]

    r = dock(nav, det["dock"], refine=refine)
    out.update({k: v for k, v in r.items() if k not in out})
    out["stage"] = "DOCK" if not r["ok"] else None
    out["panel_det"] = latest["det"]          # 이후 관찰 자세 계산에 쓰는 최신 패널 추정 (center, normal)
    return out
