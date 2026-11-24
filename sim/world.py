"""scene_v3 장면: 주행 휠체어 + 엘리베이터 홀 + 숫자 버튼 6개 + 장애물.

make_world(seed, condition) 하나로 초기 상태가 결정된다(시드만으로 재현).
에피소드 시작 '전'에만 배치를 바꾼다: 패널 위치(body_pos), 버튼 칸 배치(숫자 섞기), 장애물 위치, 휠체어 출발 자세(qpos).
에피소드 중에는 모든 움직임이 액추에이터 -> mj_step 으로만 일어난다.

조건 (일반화 축 = 패널 위치, 패널 주위 장애물)
  none      : 기본 위치, 장애물 없음, 기본 숫자 배치
  base      : 패널 벽 방향 ±0.2m / 높이 ±2cm, 장애물 0~1개(직선 경로 밖), 출발 ±0.3m ±15°
  variation : 패널 벽 방향 ±0.6m / 높이 ±6cm, 장애물 2~3개(1개는 직선 경로를 막음, 1개는 패널 주위), 출발 ±0.5m ±45°
  stress    : (한계 분석, 평가 범위 밖) 패널 ±1.0m / ±10cm, 장애물 3개(경로 1 + 패널 주위 2), 출발 ±0.5m ±60°
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from models.make_scene_v3 import (BUTTON_RGBA, DEFAULT_LAYOUT, DOCK_REL, LAMP_RGBA, N_LIDAR, OBSTACLES,
                                  PANEL_NOMINAL, PARK, SLOTS, START_NOMINAL)
from sim.scene import Sim

ROOT = Path(__file__).resolve().parents[1]
V3_XML = Path(os.environ.get("KWVIP_SCENE_V3", ROOT / "models" / "scene_v3.xml"))
BUILDING_FLOORS = (1, 2, 3, 4, 5, 6)   # 건물 층 정보(사전지식) -> 7층 등은 움직이기 전에 거절
WALL_Y = 0.0                          # 패널이 붙은 벽 앞면
HALL_X = (-0.5, 7.5)
HALL_Y = (0.0, 3.6)
FEAS_MARGIN = 0.05                    # 실행 가능성 검사에서 로봇 반경에 더하는 여유 (m)
LAMP_MM = 10.0                        # 이 변위 이상이면 '누름 등록' -> 램프 점등
LAMP = np.array(LAMP_RGBA)
BTN = np.array([float(v) for v in BUTTON_RGBA.split()])

_CACHE: dict[str, mujoco.MjModel] = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def dock_pose(panel_pos) -> np.ndarray:
    """패널 중심 -> 휠체어 도킹 자세 (x, y, yaw).  휠체어 기준 패널 상대 위치 = DOCK_REL (v2 와 같음), yaw 0(벽과 평행)."""
    return np.array([panel_pos[0] - DOCK_REL[0], panel_pos[1] - DOCK_REL[1], 0.0])


# ---------------- 무작위화 ----------------
def _sample_layout(rng, condition):
    if condition == "none":
        return tuple(DEFAULT_LAYOUT)
    return tuple(int(v) for v in rng.permutation(np.arange(1, 7)))


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _obstacle_radius(i):
    typ, size, _ = OBSTACLES[i]
    s = [float(v) for v in size.split()]
    return float(np.hypot(s[0], s[1])) if typ == "box" else s[0]


def sample_layout(rng: np.random.Generator, condition: str) -> dict:
    """패널 오프셋, 숫자 배치, 출발 자세, 장애물 배치."""
    if condition == "none":
        return {"panel_off": np.zeros(3), "layout": tuple(DEFAULT_LAYOUT), "start": np.array(START_NOMINAL),
                "obstacles": {}, "kind": "none"}
    if condition in ("base", "train"):
        pdx, pdz, sxy, syaw, n_obs = 0.2, 0.02, 0.3, np.radians(15), int(rng.integers(0, 2))
    elif condition == "variation":
        pdx, pdz, sxy, syaw, n_obs = 0.6, 0.06, 0.5, np.radians(45), int(rng.integers(2, 4))
    elif condition == "stress":   # 한계 분석용(평가 범위 밖): 패널 ±1.0m / ±10cm, 장애물 3개(경로 1 + 패널 주위 2)
        pdx, pdz, sxy, syaw, n_obs = 1.0, 0.10, 0.5, np.radians(60), 3
    else:
        raise ValueError(f"unknown condition: {condition}")
    off = np.array([rng.uniform(-pdx, pdx), 0.0, rng.uniform(-pdz, pdz)])
    layout = _sample_layout(rng, condition)
    start = np.array(START_NOMINAL) + np.array([rng.uniform(-sxy, sxy), rng.uniform(-sxy, sxy), rng.uniform(-syaw, syaw)])
    panel = np.array(PANEL_NOMINAL) + off
    dock = dock_pose(panel)
    pre = dock[:2] - np.array([0.6, 0.0])
    a, b = start[:2], pre
    order = rng.permutation(len(OBSTACLES))
    obstacles: dict[int, np.ndarray] = {}
    kinds = []
    for slot in range(n_obs):
        i = int(order[slot])
        r = _obstacle_radius(i)
        if condition in ("variation", "stress") and slot == 0:
            want = "block"      # 직선 경로 위
        elif (condition == "variation" and slot == 1) or (condition == "stress" and slot >= 1):
            want = "panel"      # 패널 주위 (도킹 구역 바깥)
        else:
            want = "off"        # 직선 경로 밖
        for _ in range(200):
            if want == "block":
                t = rng.uniform(0.35, 0.7)
                p = a + t * (b - a) + rng.uniform(-0.15, 0.15, 2)
            elif want == "panel":
                ang = rng.uniform(0, np.pi)  # 도킹 자세 주변 반원 (벽 반대쪽)
                dist = rng.uniform(0.95, 1.3) + r
                p = dock[:2] + dist * np.array([np.cos(ang), np.sin(ang)])
            else:
                p = np.array([rng.uniform(1.8, dock[0] - 0.8), rng.uniform(0.4, 3.2)])
            # 벽·출발·도킹 여유, 다른 장애물과 겹침 금지
            if not (HALL_X[0] + r + 0.1 < p[0] < HALL_X[1] - r - 0.1 and HALL_Y[0] + r + 0.05 < p[1] < HALL_Y[1] - r - 0.05):
                continue
            if np.linalg.norm(p - start[:2]) < 0.95 + r or np.linalg.norm(p - dock[:2]) < 0.9 + r:
                continue
            if want == "off" and _seg_dist(p, a, b) < 0.9 + r:
                continue
            if any(np.linalg.norm(p - q) < r + _obstacle_radius(j) + 0.15 for j, q in obstacles.items()):
                continue
            obstacles[i] = p
            kinds.append(want)
            break
    return {"panel_off": off, "layout": layout, "start": start, "obstacles": obstacles,
            "kind": f"{condition}:obs{len(obstacles)}({','.join(kinds)})"}


# ---------------- 시뮬레이터 ----------------
@dataclass
class WorldSim(Sim):
    layout: tuple = ()
    start_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obstacles: dict = field(default_factory=dict)
    n_collision_steps: int = 0        # 휠체어-벽/장애물 접촉이 있었던 물리 스텝 수
    collided_with: set = field(default_factory=set)
    lit: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        m = self.model
        self.base_qadr = np.array([m.joint(j).qposadr[0] for j in ("base_x", "base_y", "base_yaw")])
        self.base_dadr = np.array([m.joint(j).dofadr[0] for j in ("base_x", "base_y", "base_yaw")])
        self.base_act = np.array([m.actuator(a).id for a in ("base_vx", "base_vy", "base_wz")])
        self.lidar_adr = np.array([m.sensor_adr[m.sensor(f"lidar_{k:02d}").id] for k in range(N_LIDAR)])
        self.touch_adr = [m.sensor_adr[m.sensor(n).id] for n in ("touch_left", "touch_right")]
        chair = m.body("wheelchair_base").id
        self.chair_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == chair}
        self.env_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] & 4 or m.geom(g).name == "panel"}
        self.btn_geom = {i: m.geom(f"button_{i}_geom").id for i in self.floors}
        self.lit = {i: False for i in self.floors}
        self.twist = np.zeros(2)          # (v, ω) 명령
        self.hold_pose: np.ndarray | None = None
        self.hold_kp = np.array([2.0, 2.0, 2.5])

    # --- 베이스 ---
    def base_pose(self) -> np.ndarray:
        return self.data.qpos[self.base_qadr].copy()

    def base_vel(self) -> np.ndarray:
        return self.data.qvel[self.base_dadr].copy()

    def set_twist(self, v: float, w: float):
        """차동구동 명령 (전진 속도 m/s, 회전 rad/s).  자세 유지 모드를 끈다."""
        self.twist = np.array([v, w], float)
        self.hold_pose = None

    def hold(self, pose=None):
        """현재(또는 주어진) 자세를 P 제어로 유지 (누를 때 반력 대비)."""
        self.hold_pose = self.base_pose() if pose is None else np.asarray(pose, float)

    def _apply_base(self):
        th = self.data.qpos[self.base_qadr[2]]
        if self.hold_pose is not None:
            e = self.hold_pose - self.data.qpos[self.base_qadr]
            e[2] = wrap(e[2])
            # 몸체 기준 전진 성분만 허용 (차동구동): 옆 오차는 아주 작다고 보고 그대로 둔다
            fwd = np.array([np.cos(th), np.sin(th)])
            v = self.hold_kp[0] * (e[:2] @ fwd)
            w = self.hold_kp[2] * e[2]
        else:
            v, w = self.twist
        self.data.ctrl[self.base_act] = [v * np.cos(th), v * np.sin(th), w]

    # --- 센서 ---
    def lidar(self) -> np.ndarray:
        """72빔 거리 (m), 맞지 않으면 inf."""
        r = self.data.sensordata[self.lidar_adr].copy()
        r[r < 0] = np.inf
        return r

    def touch(self) -> float:
        return float(sum(self.data.sensordata[a] for a in self.touch_adr))

    # --- 진행 ---
    def step(self, n: int = 1):
        m, d = self.model, self.data
        for _ in range(n):
            self._apply_base()
            mujoco.mj_step(m, d)
            for i in self.floors:
                v = d.qpos[self.slide_adr[i]]
                if v > self.peak[i]:
                    self.peak[i] = float(v)
                if not self.lit[i] and v * 1000 >= LAMP_MM:
                    self.lit[i] = True
                    m.geom_rgba[self.btn_geom[i]] = LAMP
            hit = False
            for c in d.contact[:d.ncon]:
                g1, g2 = c.geom1, c.geom2
                if (g1 in self.chair_geoms and g2 in self.env_geoms) or (g2 in self.chair_geoms and g1 in self.env_geoms):
                    hit = True
                    self.collided_with.add(m.geom(g2 if g1 in self.chair_geoms else g1).name)
            self.n_collision_steps += hit
            for h in self.step_hooks:
                h(self)

    def judge(self, floor: int | None) -> dict:
        j = super().judge(floor)
        j["collisions"] = int(self.n_collision_steps)
        j["collided_with"] = sorted(self.collided_with)
        if self.n_collision_steps > 0:
            j["success"] = False
        return j


def _load(xml) -> mujoco.MjModel:
    key = str(Path(xml).resolve())
    if key not in _CACHE:
        _CACHE[key] = mujoco.MjModel.from_xml_path(key)
    return _CACHE[key]


def _apply(m, lay):
    # 패널
    m.body_pos[m.body("elevator_panel").id] = np.array(PANEL_NOMINAL) + lay["panel_off"]
    # 숫자 배치: 칸 k 에 층 layout[k] 버튼
    for (x, z), n in zip(SLOTS, lay["layout"]):
        m.body_pos[m.body(f"elevator_button_{n}").id] = [x, 0.065, z]
        m.geom_rgba[m.geom(f"button_{n}_geom").id] = BTN
    # 장애물
    for i in range(len(OBSTACLES)):
        bid = m.body(f"obs_{i}").id
        h = m.body_pos[bid][2]
        if i in lay["obstacles"]:
            m.body_pos[bid] = [*lay["obstacles"][i], h]
        else:
            m.body_pos[bid] = [PARK[0] + 2 * i, PARK[1], h]
    d = mujoco.MjData(m)
    qa = [m.joint(j).qposadr[0] for j in ("base_x", "base_y", "base_yaw")]
    d.qpos[qa] = lay["start"]
    mujoco.mj_forward(m, d)
    return d


def feasible(m, d, lay) -> bool:
    """정답 격자에서 출발 -> 도킹 전 지점까지 경로가 있는지 (장면 생성 시 1회)."""
    from nav.dock import pre_dock_candidates
    from nav.grid import ROBOT_R, gt_grid
    from nav.planner import astar
    r = ROBOT_R + FEAS_MARGIN  # 격자 이산화·라이다 관측 차이에 대한 여유
    g = gt_grid(m, d)
    dock = dock_pose(np.array(PANEL_NOMINAL) + lay["panel_off"])
    if g.clearance_at(dock[:2]) < ROBOT_R:
        return False
    return any(g.clearance_at(p) >= ROBOT_R and astar(g, lay["start"][:2], p, r) is not None
               for _, p in pre_dock_candidates(dock))


def make_world(seed: int, condition: str = "base", floor: int | None = None, xml=None,
               check_feasible: bool = True) -> WorldSim:
    """같은 (seed, condition) 이면 항상 같은 초기 상태. 경로가 없는 배치는 같은 rng 로 다시 뽑는다."""
    m = _load(xml or V3_XML)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        lay = sample_layout(rng, condition)
        d = _apply(m, lay)
        if not check_feasible or not lay["obstacles"] or feasible(m, d, lay):
            break
    if floor is None:
        floor = int(rng.integers(1, 7))
    arm = [m.actuator(f"joint{i}").id for i in range(1, 7)]
    d.ctrl[arm] = d.qpos[[m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]]
    d.ctrl[m.actuator("gripper").id] = 0.0
    return WorldSim(model=m, data=d, seed=seed, condition=condition, floor=floor,
                    panel_offset=lay["panel_off"], offset_kind=lay["kind"], layout=lay["layout"],
                    start_pose=lay["start"].copy(), obstacles=lay["obstacles"])
