"""목표 위치(기본): 시뮬레이터 좌표(ground truth)를 그대로 사용.

반환 형식은 camera.py 와 같다 -> perception 출력만 바꿔서 같은 평가를 다시 돌릴 수 있다.
  {"face": 버튼 앞면 중심 (world, m), "into": 누르는 방향 단위벡터, "source": "site_gt"}
"""
from __future__ import annotations

import numpy as np

BUTTON_HALF_LEN = 0.015  # cylinder half-height (scene_v2.xml)


def locate_button(sim, floor: int) -> dict:
    m, d = sim.model, sim.data
    bid = m.body(f"elevator_button_{floor}").id
    R = d.xmat[bid].reshape(3, 3)
    into = R @ np.array([0.0, -1.0, 0.0])  # slide 축(0 -1 0) = 누르는 방향
    center = d.geom(f"button_{floor}_geom").xpos.copy()
    face = center - into * BUTTON_HALF_LEN
    return {"face": face, "into": into / np.linalg.norm(into), "source": "site_gt"}


KEY_HALF_T = 0.004  # 도어락 키 반두께 (scene_v4.xml)


def locate_key(sim, key: str) -> dict:
    """scene_v4 도어락 키 앞면 중심(시뮬레이터 좌표, 기준선용)."""
    from models.make_scene_v4 import KEY_NAME
    m, d = sim.model, sim.data
    bid = m.body(f"key_{KEY_NAME[key]}").id
    R = d.xmat[bid].reshape(3, 3)
    into = R @ np.array([0.0, -1.0, 0.0])
    face = d.xpos[bid] - into * KEY_HALF_T
    return {"face": face, "into": into / np.linalg.norm(into), "source": "site_gt"}
