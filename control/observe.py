"""관찰 자세: 손목 카메라가 키패드를 OBS_DIST 앞에서 보도록 팔을 움직이고 키 글자를 읽는다 (scene_v4).

정면(시선 = -n)으로 물러난 자세는 팔 마운트와 가까워 닿지 않을 수 있어 '비스듬히' 보는 시점도 후보로 둔다:
시선을 휠체어 앞쪽으로 yaw, 아래로 tilt 기울이고 라벨 면 중심에서 dist. 비스듬한 시선은 perception/digits.rectify 의
정면화가 없앤다. 후보 (yaw, tilt, 거리)를 순서대로 시도해 처음 닿는 시점을 쓴다.
"""
from __future__ import annotations

import numpy as np

from control.press_fsm import PressConfig, PressExecutor
from models.make_scene_v4 import KEY_FACE
from perception.digits import CAM_AHEAD, OBS_DIST, PAD_UP, read_keypad

VIEWS = [(15, 10, OBS_DIST), (25, 10, OBS_DIST), (15, 25, OBS_DIST), (0, 10, OBS_DIST),
         (30, 20, 0.10), (15, 10, 0.15)]


def observe_pose_target(origin, normal, yaw_deg=15, tilt_deg=10, dist=OBS_DIST, aim=0.0):
    """(press_tip 목표, 접근축(시선), 키패드 라벨 면 중심).  hz = z x n (도어락 가로축).
    aim: 시선 중심을 위(+)/아래(-)로 옮김 (m). VERIFY 에서 상태 LED 를 화면 가운데에 둘 때 -0.06."""
    n = np.asarray(normal, float)
    hz = np.cross([0, 0, 1.0], n)
    hz /= np.linalg.norm(hz)
    zu = np.array([0, 0, 1.0])
    y, t = np.radians(yaw_deg), np.radians(tilt_deg)
    d = -np.cos(y) * np.cos(t) * n - np.sin(y) * np.cos(t) * hz - np.sin(t) * zu
    label = np.asarray(origin, float) + KEY_FACE * n + (PAD_UP + aim) * zu
    return label - (dist + CAM_AHEAD) * d, d, label


def observe_panel(sim, wcam, origin, normal, ex: PressExecutor | None = None, t_move=2.5, aim=0.0) -> dict:
    ex = ex or PressExecutor(sim, PressConfig(timeout_s=60.0))
    ex.ik.sync(sim.data)
    for view in VIEWS:   # 닿는 첫 시점
        tip, into, _ = observe_pose_target(origin, normal, *view, aim=aim)
        r = ex.ik.solve(tip, into, q_seed=sim.arm_q(), rng=np.random.default_rng(0), n_starts=20)
        if r.ok(0.005, 4.0):
            break
    else:
        return {"ok": False, "why": "observe_ik", "ik_err_mm": round(r.pos_err * 1000, 1)}
    ex.stage("OBSERVE")
    ex.move_ctrl(r.q, t_move, grip=0.0)
    sim.step(int(0.4 / sim.dt))          # 흔들림 가라앉힘
    ex.stage("READ_KEYS")
    read = read_keypad(wcam, sim.data, origin, np.asarray(normal, float))
    return {"ok": True, "why": None, "read": read, "q_obs": r.q, "view": view}
