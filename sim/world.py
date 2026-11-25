"""scene_v4 장면: 주행 휠체어 + 아파트 복도 + 현관문 도어락(3x4 키패드) + 고정 장애물(택배·사람).

make_world(seed, condition) 하나로 초기 상태가 결정된다(시드만으로 재현).
에피소드 시작 '전'에만 배치를 바꾼다: 현관문 위치·도어락 높이(body_pos), 장애물 위치, 휠체어 출발 자세(qpos),
등록 비밀번호(도어락과 로봇 설정에 같이 등록되는 값). 에피소드 중 움직임은 모두 액추에이터 -> mj_step 으로만 일어난다.

조건 (일반화 축 = 도어락 위치, 패널 주위 장애물)
  none      : 기본 위치, 장애물 없음, 비밀번호 2580 (회귀 점검용)
  base      : 문 x ±0.3m, 도어락 높이 1.08~1.14m, 장애물 2~3개, 출발 ±0.3m ±15°
  variation : 문 x ±0.6m, 도어락 높이 LOCK_Z_RANGE 전체, 도어락 가로 ±3cm, 장애물 2~3개, 출발 ±0.5m ±45°
  stress    : (한계 분석, 평가 범위 밖) 도어락 높이 0.92~1.32m(팔 도달 범위 밖 포함), 장애물 3개, 출발 ±60°
장애물: 첫째는 '패널 쪽'(도킹 자세 앞 접근로 근처: 우회해서 들어가야 함), 둘째는 출발→도어락 직선 위, 셋째(있으면) 경로 밖.
도킹 자세·키 접근은 막지 않는다 (도킹 자리 여유 + 실행 가능성 검사로 보장).

도어락 동작 (sim 이 흉내 냄): 키 변위 >= REG_MM 이면 '입력' 1회 (떼면 다시 입력 가능), 누르는 동안 키 백라이트.
  '*' = 입력 지움, '#' = 확인: 입력이 등록 비밀번호와 같으면 열림(LED 초록), 다르면 틀림(LED 빨강) 후 입력 지움.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from models.make_scene_v4 import (DOCK_REL, DOOR_NOMINAL_X, KEY_LIT_RGBA, KEY_NAME, KEY_RGBA, KEYS, LED_IDLE,
                                  LED_OPEN, LED_WRONG, LOCK_REL_NOMINAL, N_LIDAR, OBSTACLES, PARK, START_NOMINAL)
from sim.scene import Sim

ROOT = Path(__file__).resolve().parents[1]
V4_XML = Path(os.environ.get("KWVIP_SCENE_V4", ROOT / "models" / "scene_v4.xml"))
DEFAULT_PASSWORD = "2580"
LOCK_Z_RANGE = (1.02, 1.20)          # 도어락 원점 높이 범위 (m): 1.00~1.22 에서 키 전부 입력 확인 (tests/check_scene_v4.py) - 여유 2cm
HALL_X = (-0.5, 7.5)
HALL_Y = (0.0, 3.6)
FEAS_MARGIN = 0.05
REG_MM = 2.5                         # 키 입력 등록 변위 (mm)
RELEASE_MM = 1.0
KEY_LIT = np.array(KEY_LIT_RGBA)
KEY_DARK = np.array([float(v) for v in KEY_RGBA.split()])

_CACHE: dict[str, mujoco.MjModel] = {}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def dock_pose(lock_pos) -> np.ndarray:
    """도어락 원점(월드) -> 휠체어 도킹 자세 (x, y, yaw=0, 벽과 평행)."""
    return np.array([lock_pos[0] - DOCK_REL[0], lock_pos[1] - DOCK_REL[1], 0.0])


def lock_world(door_x, lock_rel) -> np.ndarray:
    return np.array([door_x + lock_rel[0], lock_rel[1], lock_rel[2]])


# ---------------- 무작위화 ----------------
def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _obstacle_radius(i):
    kind, size, _ = OBSTACLES[i]
    s = [float(v) for v in size.split()]
    return float(np.hypot(s[0], s[1])) if kind == "parcel" else s[0] + 0.07   # 사람: 어깨 폭 포함


def sample_layout(rng: np.random.Generator, condition: str) -> dict:
    """현관문 x, 도어락 상대 위치, 등록 비밀번호, 출발 자세, 장애물 배치."""
    if condition == "none":
        return {"door_x": DOOR_NOMINAL_X, "lock_rel": np.array(LOCK_REL_NOMINAL), "password": DEFAULT_PASSWORD,
                "start": np.array(START_NOMINAL), "obstacles": {}, "kind": "none"}
    if condition in ("base", "train"):
        ddx, zr, lx, sxy, syaw = 0.3, (1.08, 1.14), 0.0, 0.3, np.radians(15)
    elif condition == "variation":
        ddx, zr, lx, sxy, syaw = 0.6, LOCK_Z_RANGE, 0.03, 0.5, np.radians(45)
    elif condition == "stress":   # 한계 분석(평가 범위 밖): 팔 도달 범위(1.00~1.22m)를 넘는 높이 포함, 장애물 3개, 출발 ±60°
        ddx, zr, lx, sxy, syaw = 0.6, (0.92, 1.32), 0.03, 0.5, np.radians(60)
    else:
        raise ValueError(f"unknown condition: {condition}")
    door_x = DOOR_NOMINAL_X + rng.uniform(-ddx, ddx)
    lock_rel = np.array(LOCK_REL_NOMINAL) + np.array([rng.uniform(-lx, lx), 0.0, 0.0])
    lock_rel[2] = rng.uniform(*zr)
    password = "".join(str(int(d)) for d in rng.integers(0, 10, int(rng.integers(4, 7))))
    start = np.array(START_NOMINAL) + np.array([rng.uniform(-sxy, sxy), rng.uniform(-sxy, sxy), rng.uniform(-syaw, syaw)])
    dock = dock_pose(lock_world(door_x, lock_rel))
    pre = dock[:2] - np.array([0.6, 0.0])
    a, b = start[:2], pre
    n_obs = 3 if condition == "stress" else int(rng.integers(2, 4))
    order = rng.permutation(len(OBSTACLES))
    obstacles: dict[int, np.ndarray] = {}
    kinds = []
    for slot in range(n_obs):
        i = int(order[slot])
        r = _obstacle_radius(i)
        want = ("panel", "block", "off")[slot]
        for _ in range(300):
            if want == "panel":       # 도킹 자세 앞쪽 접근로 근처 (패널 자체·도킹 자리는 비움)
                ang = rng.uniform(2.6, 3.7)
                u = np.array([np.cos(ang), np.sin(ang)])
                p = dock[:2] + (rng.uniform(0.95, 1.35) + r) * u
            elif want == "block":     # 출발 -> 도어락 직선 위
                p = a + rng.uniform(0.35, 0.7) * (b - a) + rng.uniform(-0.15, 0.15, 2)
            else:
                p = np.array([rng.uniform(1.6, dock[0] - 0.8), rng.uniform(0.4, 3.2)])
            if not (HALL_X[0] + r + 0.1 < p[0] < HALL_X[1] - r - 0.1 and HALL_Y[0] + r + 0.05 < p[1] < HALL_Y[1] - r - 0.05):
                continue
            if np.linalg.norm(p - start[:2]) < 0.95 + r or np.linalg.norm(p - dock[:2]) < 0.9 + r:
                continue
            if want == "off" and _seg_dist(p, a, b) < 0.9 + r:
                continue
            if any(np.linalg.norm(p - q) < r + _obstacle_radius(j) + 0.15 for j, q in obstacles.items()):
                continue
            obstacles[i] = p
            kinds.append(f"{want}:{OBSTACLES[i][0]}")
            break
    return {"door_x": door_x, "lock_rel": lock_rel, "password": password, "start": start, "obstacles": obstacles,
            "kind": f"{condition}:obs{len(obstacles)}({','.join(kinds)})"}


# ---------------- 시뮬레이터 ----------------
@dataclass
class WorldSim(Sim):
    password: str = DEFAULT_PASSWORD
    start_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    obstacles: dict = field(default_factory=dict)
    door_x: float = DOOR_NOMINAL_X
    n_collision_steps: int = 0        # 휠체어-벽/장애물 접촉이 있었던 물리 스텝 수
    collided_with: set = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()
        m = self.model
        self.keys = list(KEYS)
        self.key_qadr = {k: m.joint(f"key_{KEY_NAME[k]}_slide").qposadr[0] for k in self.keys}
        self.key_geom = {k: m.geom(f"key_{KEY_NAME[k]}_geom").id for k in self.keys}
        self.led_geom = m.geom("lock_led").id
        self.key_peak = {k: 0.0 for k in self.keys}
        self.key_down = {k: False for k in self.keys}
        self.events: list[tuple[str, float]] = []      # (키, 시각) 입력 순서
        self.buffer = ""
        self.lock_state = "idle"                       # idle | open | wrong
        self.lock_log: list[tuple[str, str, float]] = []
        self.base_qadr = np.array([m.joint(j).qposadr[0] for j in ("base_x", "base_y", "base_yaw")])
        self.base_dadr = np.array([m.joint(j).dofadr[0] for j in ("base_x", "base_y", "base_yaw")])
        self.base_act = np.array([m.actuator(a).id for a in ("base_vx", "base_vy", "base_wz")])
        self.lidar_adr = np.array([m.sensor_adr[m.sensor(f"lidar_{k:02d}").id] for k in range(N_LIDAR)])
        self.touch_adr = m.sensor_adr[m.sensor("touch_tip").id]
        chair = m.body("wheelchair_base").id
        self.chair_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == chair}
        self.env_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] & 4 or m.geom(g).name == "lock_housing"}
        self.twist = np.zeros(2)
        self.hold_pose: np.ndarray | None = None
        self.hold_kp = np.array([2.0, 2.0, 2.5])
        for k in self.keys:
            m.geom_rgba[self.key_geom[k]] = KEY_DARK
        m.geom_rgba[self.led_geom] = LED_IDLE

    # --- 도어락 ---
    def lock_pos(self) -> np.ndarray:
        return self.data.xpos[self.model.body("doorlock").id].copy()

    def key_disp(self, k) -> float:
        return float(self.data.qpos[self.key_qadr[k]])

    def _doorlock_logic(self):
        m, d = self.model, self.data
        for k in self.keys:
            mm = d.qpos[self.key_qadr[k]] * 1000
            if mm > self.key_peak[k] * 1000:
                self.key_peak[k] = mm / 1000
            if not self.key_down[k] and mm >= REG_MM:
                self.key_down[k] = True
                m.geom_rgba[self.key_geom[k]] = KEY_LIT
                self.events.append((k, round(float(d.time), 3)))
                if k == "*":
                    self.buffer = ""
                elif k == "#":
                    ok = self.buffer == self.password
                    self.lock_state = "open" if ok else "wrong"
                    m.geom_rgba[self.led_geom] = LED_OPEN if ok else LED_WRONG
                    self.lock_log.append(("#", self.lock_state, round(float(d.time), 3)))
                    self.buffer = ""
                else:
                    self.buffer += k
            elif self.key_down[k] and mm < RELEASE_MM:
                self.key_down[k] = False
                m.geom_rgba[self.key_geom[k]] = KEY_DARK

    def entered(self) -> str:
        return "".join(k for k, _ in self.events)

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
            fwd = np.array([np.cos(th), np.sin(th)])
            v = self.hold_kp[0] * (e[:2] @ fwd)
            w = self.hold_kp[2] * e[2]
        else:
            v, w = self.twist
        self.data.ctrl[self.base_act] = [v * np.cos(th), v * np.sin(th), w]

    # --- 센서 ---
    def lidar(self) -> np.ndarray:
        r = self.data.sensordata[self.lidar_adr].copy()
        r[r < 0] = np.inf
        return r

    def touch(self) -> float:
        return float(self.data.sensordata[self.touch_adr])

    # --- 진행 ---
    def step(self, n: int = 1):
        m, d = self.model, self.data
        for _ in range(n):
            self._apply_base()
            mujoco.mj_step(m, d)
            self._doorlock_logic()
            hit = False
            for c in d.contact[:d.ncon]:
                g1, g2 = c.geom1, c.geom2
                if (g1 in self.chair_geoms and g2 in self.env_geoms) or (g2 in self.chair_geoms and g1 in self.env_geoms):
                    hit = True
                    self.collided_with.add(m.geom(g2 if g1 in self.chair_geoms else g1).name)
            self.n_collision_steps += hit
            for h in self.step_hooks:
                h(self)

    def judge(self, password: str | None) -> dict:
        """password 가 있으면: 마지막 '*' 이후 입력이 password + '#' 이고 도어락이 열렸고 충돌 0 이면 성공.
        None(움직이면 안 되는 명령)이면: 키 입력 0, 모든 키 최대 변위 < 1mm, 충돌 0 이면 '정상(미동작)'."""
        mm = {k: round(self.key_peak[k] * 1000, 2) for k in self.keys}
        seq = self.entered()
        out = {"entered": seq, "lock_state": self.lock_state, "unlocked": self.lock_state == "open",
               "peaks_mm": mm, "collisions": int(self.n_collision_steps), "collided_with": sorted(self.collided_with)}
        if password is None:
            other = max(mm.values())
            return {**out, "success": not seq and other < 1.0 and self.n_collision_steps == 0,
                    "peak_mm": 0.0, "other_mm": other, "target": ""}
        target = password + "#"
        used = set(target)
        out.update({"target": target, "peak_mm": min(mm[k] for k in used),
                    "other_mm": max([mm[k] for k in self.keys if k not in used] or [0.0])})
        # '*'(지움)로 틀린 입력을 지우고 다시 넣은 경우도 성공 (실제 도어락 사용법). 틀린 키 입력 수는 따로 기록.
        out["wrong_inputs"] = sum(1 for k in seq if k not in used)
        out["success"] = bool(seq.split("*")[-1] == target and self.lock_state == "open" and self.n_collision_steps == 0)
        return out


def _load(xml) -> mujoco.MjModel:
    key = str(Path(xml).resolve())
    if key not in _CACHE:
        _CACHE[key] = mujoco.MjModel.from_xml_path(key)
    return _CACHE[key]


def _apply(m, lay):
    m.body_pos[m.body("home_door").id] = [lay["door_x"], 0.0, 0.0]
    m.body_pos[m.body("doorlock").id] = lay["lock_rel"]
    for i in range(len(OBSTACLES)):
        bid = m.body(f"obs_{i}").id
        h = m.body_pos[bid][2]
        m.body_pos[bid] = [*lay["obstacles"][i], h] if i in lay["obstacles"] else [PARK[0] + 2 * i, PARK[1], h]
    d = mujoco.MjData(m)
    d.qpos[[m.joint(j).qposadr[0] for j in ("base_x", "base_y", "base_yaw")]] = lay["start"]
    mujoco.mj_forward(m, d)
    return d


def feasible(m, d, lay) -> bool:
    """정답 격자에서 출발 -> 도킹 전 지점까지 경로가 있는지 + 도킹 자리가 비었는지 (장면 생성 시 1회)."""
    from nav.dock import pre_dock_candidates
    from nav.grid import ROBOT_R, gt_grid
    from nav.planner import astar
    r = ROBOT_R + FEAS_MARGIN
    g = gt_grid(m, d)
    dock = dock_pose(lock_world(lay["door_x"], lay["lock_rel"]))
    if g.clearance_at(dock[:2]) < ROBOT_R + 0.1:
        return False
    return any(g.clearance_at(p) >= ROBOT_R and astar(g, lay["start"][:2], p, r) is not None
               for _, p in pre_dock_candidates(dock))


def make_world(seed: int, condition: str = "base", password: str | None = None, xml=None,
               check_feasible: bool = True, floor=None) -> WorldSim:
    """같은 (seed, condition) 이면 항상 같은 초기 상태. 경로가 없는 배치는 같은 rng 로 다시 뽑는다.
    password: 주면 시드의 등록 비밀번호 대신 이것을 등록 (floor 는 이전 인터페이스 호환용, 쓰지 않음)."""
    m = _load(xml or V4_XML)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        lay = sample_layout(rng, condition)
        d = _apply(m, lay)
        if not check_feasible or not lay["obstacles"] or feasible(m, d, lay):
            break
    arm = [m.actuator(f"joint{i}").id for i in range(1, 7)]
    d.ctrl[arm] = d.qpos[[m.joint(f"joint{i}").qposadr[0] for i in range(1, 7)]]
    d.ctrl[m.actuator("gripper").id] = 0.0
    return WorldSim(model=m, data=d, seed=seed, condition=condition, floor=None,
                    panel_offset=np.array([lay["door_x"] - DOOR_NOMINAL_X, 0.0, lay["lock_rel"][2] - LOCK_REL_NOMINAL[2]]),
                    offset_kind=lay["kind"], password=password or lay["password"], start_pose=lay["start"].copy(),
                    obstacles=lay["obstacles"], door_x=lay["door_x"])
