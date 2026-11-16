"""장면 로드와 무작위화.

make_scene(seed, condition) 하나로 초기 상태가 결정된다(시드만으로 재현).
  - 패널 위치는 model.body_pos 로 옮긴다(순간이동이 아니라 '장면 배치'의 변경; 에피소드 시작 전에만 수행).
  - 목표 층은 floor 인자로 주거나, 없으면 같은 rng 에서 뽑는다.

조건(condition) 정의는 docs/kw-vip과제_구현개발프로세스.md 의 '평가 조건 정의'를 따른다.
  train / base : x ±2cm, z ±2cm (균등)
  variation    : x U(-6, +3)cm, 그리고 [z ±U(5, 8)cm] 또는 [패널이 U(4, 8)cm 멀어짐(y -)] 중 하나
  none         : 오프셋 없음 (기본 위치)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = Path(os.environ.get("KWVIP_SCENE", ROOT / "models" / "scene_v2.xml"))
FLOORS = (1, 2, 3)
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

_MODEL_CACHE: dict[str, tuple[mujoco.MjModel, np.ndarray]] = {}


def sample_panel_offset(rng: np.random.Generator, condition: str) -> tuple[np.ndarray, str]:
    """조건별 패널 오프셋 [dx, dy, dz] (m) 와 설명 문자열."""
    if condition in ("train", "base"):
        off = np.array([rng.uniform(-0.02, 0.02), 0.0, rng.uniform(-0.02, 0.02)])
        return off, "xz±2cm"
    if condition == "variation":
        dx = rng.uniform(-0.06, 0.03)
        if rng.random() < 0.5:
            dz = rng.choice([-1.0, 1.0]) * rng.uniform(0.05, 0.08)
            return np.array([dx, 0.0, dz]), "z±5~8cm"
        dy = -rng.uniform(0.04, 0.08)  # 패널 법선이 +y(사용자 쪽)이므로 -y 가 '멀어짐'
        return np.array([dx, dy, 0.0]), "far4~8cm"
    if condition == "none":
        return np.zeros(3), "none"
    raise ValueError(f"unknown condition: {condition}")


def _load(xml: str | Path) -> tuple[mujoco.MjModel, np.ndarray]:
    key = str(Path(xml).resolve())
    if key not in _MODEL_CACHE:
        m = mujoco.MjModel.from_xml_path(key)
        _MODEL_CACHE[key] = (m, m.body_pos[m.body("elevator_panel").id].copy())
    return _MODEL_CACHE[key]


@dataclass
class Sim:
    model: mujoco.MjModel
    data: mujoco.MjData
    seed: int
    condition: str
    floor: int | None
    panel_offset: np.ndarray
    offset_kind: str
    peak: dict[int, float] = field(default_factory=dict)  # 버튼별 최대 변위 (m)
    step_hooks: list = field(default_factory=list)  # f(sim) 를 매 물리 스텝 후 호출 (녹화 등)

    def __post_init__(self):
        m = self.model
        self.slide_adr = {i: m.joint(f"button_{i}_slide").qposadr[0] for i in FLOORS}
        self.arm_qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.grip_qadr = m.joint("joint7").qposadr[0]
        self.peak = {i: 0.0 for i in FLOORS}
        self.dt = m.opt.timestep

    # --- 상태 ---
    @property
    def time(self) -> float:
        return float(self.data.time)

    def button_disp(self, i: int) -> float:
        return float(self.data.qpos[self.slide_adr[i]])

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def joint_state7(self) -> np.ndarray:
        """관절 상태 7차원 (joint1~6 + 그리퍼 joint7)."""
        return np.r_[self.arm_q(), self.data.qpos[self.grip_qadr]]

    # --- 진행 ---
    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            for i in FLOORS:
                v = self.data.qpos[self.slide_adr[i]]
                if v > self.peak[i]:
                    self.peak[i] = float(v)
            for h in self.step_hooks:
                h(self)

    def judge(self, floor: int | None) -> dict:
        """성공 판정: 목표 버튼 최대 변위 >= 10mm 이고 다른 버튼 < 3mm.
        floor=None(움직이면 안 되는 명령)이면 모든 버튼 < 3mm 일 때 '정상(미동작)'."""
        mm = {i: self.peak[i] * 1000 for i in FLOORS}
        if floor is None:
            other = max(mm.values())
            return {"success": other < 3.0, "peak_mm": 0.0, "other_mm": round(other, 2), "peaks_mm": mm}
        tgt = mm[floor]
        other = max(v for i, v in mm.items() if i != floor)
        return {"success": bool(tgt >= 10.0 and other < 3.0), "peak_mm": round(tgt, 2),
                "other_mm": round(other, 2), "peaks_mm": mm}


def make_scene(seed: int, condition: str = "train", floor: int | None = None,
               xml: str | Path | None = None) -> Sim:
    """시드와 조건으로 장면을 만든다. 같은 (seed, condition) 이면 항상 같은 초기 상태."""
    m, p0 = _load(xml or DEFAULT_XML)
    rng = np.random.default_rng(seed)
    off, kind = sample_panel_offset(rng, condition)
    if floor is None:
        floor = int(rng.integers(1, 4))
    m.body_pos[m.body("elevator_panel").id] = p0 + off
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # 위치 서보 목표 = 현재 자세, 그리퍼 0(닫힘)에서 시작 -> 시작 직후 팔이 튀지 않음
    d.ctrl[:6] = d.qpos[[m.joint(j).qposadr[0] for j in ARM_JOINTS]]
    d.ctrl[6] = 0.0
    return Sim(model=m, data=d, seed=seed, condition=condition, floor=floor,
               panel_offset=off, offset_kind=kind)
