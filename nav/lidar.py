"""2D 라이다(rangefinder 72빔) -> 월드 좌표 점.

자기 위치(odometry)는 베이스 관절 qpos 를 그대로 쓴다(바퀴 미끄러짐 없는 이상적 주행계 가정, 발표에서 명시).
"""
from __future__ import annotations

import numpy as np

from models.make_scene_v3 import LIDAR_POS, N_LIDAR

ANGLES = 2 * np.pi * np.arange(N_LIDAR) / N_LIDAR
MAX_RANGE = 6.0


def lidar_origin(pose) -> np.ndarray:
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    return np.array([x + c * LIDAR_POS[0] - s * LIDAR_POS[1], y + s * LIDAR_POS[0] + c * LIDAR_POS[1]])


def scan_points(sim, max_range: float = MAX_RANGE):
    """(맞은 점들 Nx2, 원점, 거리 72) 반환. 거리가 max_range 넘는 빔은 점에서 뺀다."""
    pose = sim.base_pose()
    r = sim.lidar()
    o = lidar_origin(pose)
    a = ANGLES + pose[2]
    ok = np.isfinite(r) & (r < max_range)
    pts = o + np.stack([np.cos(a[ok]), np.sin(a[ok])], 1) * r[ok, None]
    return pts, o, r


def sector_min(r: np.ndarray, center_deg: float = 0.0, half_deg: float = 30.0) -> float:
    """몸체 기준 방향 center_deg ±half_deg 안의 최소 거리."""
    d = (np.degrees(ANGLES) - center_deg + 180) % 360 - 180
    sel = np.abs(d) <= half_deg
    return float(np.min(r[sel])) if sel.any() else np.inf


# 몸체 기준(라이다 원점 ≈ 베이스 원점) 휠체어 평면 윤곽:
#   앞쪽(발판·캐스터) x 0.30~0.60 에서 |y| <= 0.33,  가운데(팔 마운트 판) x <= 0.30 에서 |y| <= 0.46
FRONT_X, FRONT_HALF_W, MID_X, HALF_W = 0.54, 0.33, 0.30, 0.48


def front_blocked(r: np.ndarray, margin: float = 0.15, sign: float = 1.0) -> bool:
    """진행 방향(sign=+1 전진, -1 후진)으로 몸체 윤곽 바로 앞 margin 깊이 안에 라이다 점이 있는가."""
    ok = np.isfinite(r)
    x, y = r[ok] * np.cos(ANGLES[ok]) * sign, np.abs(r[ok] * np.sin(ANGLES[ok]))
    front = FRONT_X if sign > 0 else 0.55   # 뒤쪽 끝(뒷바퀴) ~0.55
    nose = (x > MID_X) & (x < front + margin) & (y < FRONT_HALF_W + 0.05)
    mid = (x > 0) & (x <= MID_X) & (y < HALF_W) & (x > MID_X - margin)
    return bool(np.any(nose | mid))
