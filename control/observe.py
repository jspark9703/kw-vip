"""관찰 자세: 손목 카메라가 패널 정면 OBS_DIST 앞에서 패널을 보도록 팔을 움직이고 숫자를 읽는다.

정면(시선 = -n)으로 20cm 이상 물러난 자세는 팔 마운트와 너무 가까워 닿지 않는다(IK 오차 2~9cm).
그래서 '비스듬히' 본다: 시선을 휠체어 앞쪽으로 yaw 20°, 아래로 tilt 15° 기울이고 라벨 면 중심에서 0.18m.
비스듬한 시선은 perception/digits.rectify 의 정면화가 없앤다. 후보 (yaw, tilt, 거리)를 순서대로 시도.
"""
from __future__ import annotations

import numpy as np

from control.press_fsm import PressConfig, PressExecutor
from perception.digits import CAM_AHEAD, LABEL_OFF, OBS_DIST, read_panel
from perception.panel import PANEL_HALF_T


VIEWS = [(20, 15, 0.18), (20, 0, 0.16), (20, 30, 0.16), (30, 15, 0.16), (20, 15, 0.13), (0, 15, 0.10)]


def observe_pose_target(panel_center, normal, yaw_deg=20, tilt_deg=15, dist=OBS_DIST):
    """(press_tip 목표, 접근축(시선), 패널 앞면 중심).  hz = z x n (패널 가로축)."""
    n = np.asarray(normal, float)
    hz = np.cross([0, 0, 1.0], n)
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    d = -np.cos(y) * np.cos(t) * n - np.sin(y) * np.cos(t) * hz - np.sin(t) * np.array([0, 0, 1.0])
    face = np.asarray(panel_center, float) + PANEL_HALF_T * n
    label = face + LABEL_OFF * n
    return label - (dist + CAM_AHEAD) * d, d, face


def observe_panel(sim, wcam, panel_center, normal, ex: PressExecutor | None = None, t_move=2.5) -> dict:
    ex = ex or PressExecutor(sim, PressConfig(timeout_s=60.0))
    ex.ik.sync(sim.data)
    for view in VIEWS:   # 닿는 첫 시점
        tip, into, face = observe_pose_target(panel_center, normal, *view)
        r = ex.ik.solve(tip, into, q_seed=sim.arm_q(), rng=np.random.default_rng(0), n_starts=20)
        if r.ok(0.005, 4.0):
            break
    else:
        return {"ok": False, "why": "observe_ik", "ik_err_mm": round(r.pos_err * 1000, 1)}
    ex.stage("OBSERVE")
    ex.move_ctrl(r.q, t_move, grip=0.0)
    sim.step(int(0.4 / sim.dt))          # 흔들림 가라앉힘
    ex.stage("READ_DIGITS")
    read = read_panel(wcam, sim.data, face, np.asarray(normal, float))
    return {"ok": True, "why": None, "read": read, "q_obs": r.q, "view": view}
