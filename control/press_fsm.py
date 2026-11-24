"""버튼 누르기 상태 기계 (IK 기준선 실행부).

상태: CLOSE_GRIPPER -> PLAN -> APPROACH(5cm 앞) -> PRESS(16mm 를 8단계로 전진) -> RETRACT -> JUDGE
모든 동작은 위치 서보 ctrl 을 smoothstep 으로 보간해서 내보내고, MuJoCo 가 물리 스텝을 진행한다.

인터페이스 (팀 합의 고정):
    execute(task, sim, ...) -> {success, peak_mm, other_mm, fail_type, t_task, ...}
fail_type: None | "perception" | "ik" | "timeout" | "contact" | "wrong_button"
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from control.ik import PressIK
from perception.site_gt import locate_button


@dataclass
class PressConfig:
    approach_dist: float = 0.05   # 버튼 앞면에서 접근 자세까지 거리 (m)
    press_depth: float = 0.016    # 앞면 기준 최대 전진 깊이 (m)
    n_press_steps: int = 8
    t_close: float = 0.3          # 그리퍼 닫기 (s)
    t_approach: float = 3.0
    t_press_step: float = 0.4
    t_retract: float = 1.0
    t_idle: float = 2.0           # 움직이면 안 되는 명령일 때 관찰 시간
    approach_tol: float = 0.010   # 접근 후 press_tip 추종 오차 허용 (m)
    settle_max: float = 1.0       # 추종 오차가 남을 때 추가 대기 한도 (s) -> 넘으면 timeout
    timeout_s: float = 15.0       # 작업 전체 시뮬레이션 시간 한도
    ik_pos_tol: float = 0.003
    ik_ang_tol: float = 3.0
    ik_step_tol: float = 0.004
    ik_starts: int = 30


class _Timeout(Exception):
    pass


def _smooth(s):
    return 3 * s * s - 2 * s ** 3


class PressExecutor:
    def __init__(self, sim, cfg: PressConfig | None = None, status=None):
        self.sim = sim
        self.cfg = cfg or PressConfig()
        self.ik = PressIK(sim.model)
        self.status = status or (lambda **kw: None)  # 녹화 오버레이용 콜백
        self.t0 = sim.time
        self.log: list[dict] = []

    # ---------------- 저수준 ----------------
    def _check_timeout(self):
        if self.sim.time - self.t0 > self.cfg.timeout_s:
            raise _Timeout()

    def move_ctrl(self, q_goal, T, grip=None):
        d, ia, ig = self.sim.data, self.sim.arm_act, self.sim.grip_act
        q0 = d.ctrl[ia].copy()
        g0 = d.ctrl[ig]
        n = max(1, int(round(T / self.sim.dt)))
        for k in range(n):
            s = _smooth((k + 1) / n)
            d.ctrl[ia] = (1 - s) * q0 + s * q_goal
            if grip is not None:
                d.ctrl[ig] = (1 - s) * g0 + s * grip
            self.sim.step()
            if k % 50 == 0:
                self._check_timeout()

    def wait_until_tracked(self, p_goal) -> bool:
        """press_tip 이 p_goal 근처(approach_tol)에 들어올 때까지 최대 settle_max 대기."""
        t_end = self.sim.time + self.cfg.settle_max
        tip = self.sim.model.site("press_tip").id
        while True:
            err = np.linalg.norm(self.sim.data.site_xpos[tip] - p_goal)
            if err <= self.cfg.approach_tol:
                return True
            if self.sim.time >= t_end:
                return False
            self.sim.step(10)

    def stage(self, name, **info):
        self.log.append({"stage": name, "t": round(self.sim.time - self.t0, 3), **info})
        self.status(state=name)

    # ---------------- 계획 ----------------
    def plan(self, face, into, rng):
        c = self.cfg
        pre = face - c.approach_dist * into
        self.ik.sync(self.sim.data)  # 현재 베이스 자세 기준으로 IK
        r = self.ik.solve(pre, into, q_seed=self.sim.arm_q(), rng=rng, n_starts=c.ik_starts)
        if not r.ok(c.ik_pos_tol, c.ik_ang_tol):
            return None, {"why": "approach_unreachable", "pos_err_mm": r.pos_err * 1e3, "ang_err_deg": r.ang_err_deg}
        way = [r.q]
        for dep in np.linspace(0, c.press_depth, c.n_press_steps + 1)[1:]:
            rs = self.ik.solve_from(face + dep * into, into, way[-1], iters=200)  # 이전 해에서 이어 풀기
            if rs.pos_err > c.ik_step_tol:
                return None, {"why": "press_path_unreachable", "depth_mm": dep * 1e3, "pos_err_mm": rs.pos_err * 1e3}
            way.append(rs.q)
        return way, {"pos_err_mm": round(r.pos_err * 1e3, 2), "ang_err_deg": round(r.ang_err_deg, 2)}

    # ---------------- 실행 ----------------
    def run_press(self, floor: int, locate=locate_button, rng=None) -> dict:
        c, sim = self.cfg, self.sim
        rng = rng if rng is not None else np.random.default_rng(sim.seed)
        out = {"fail_type": None, "perception": None, "ik": None}
        try:
            self.stage("CLOSE_GRIPPER")
            self.move_ctrl(sim.data.ctrl[sim.arm_act].copy(), c.t_close, grip=0.0)

            self.stage("PERCEIVE")
            tgt = locate(sim, floor)
            if tgt is None or tgt.get("face") is None:
                out["fail_type"] = "perception"
                out["perception"] = (tgt or {}).get("info")
                return self._finish(floor, out)
            out["perception"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in tgt.items()}
            face, into = np.asarray(tgt["face"]), np.asarray(tgt["into"])

            self.stage("PLAN")
            way, info = self.plan(face, into, rng)
            out["ik"] = info
            if way is None:
                out["fail_type"] = "ik"
                return self._finish(floor, out)

            self.stage("APPROACH")
            self.move_ctrl(way[0], c.t_approach)
            if not self.wait_until_tracked(face - c.approach_dist * into):
                out["fail_type"] = "timeout"
                return self._finish(floor, out)

            self.stage("PRESS")
            for q in way[1:]:
                self.move_ctrl(q, c.t_press_step)

            self.stage("RETRACT")
            self.move_ctrl(way[0], c.t_retract)
        except _Timeout:
            out["fail_type"] = "timeout"
        return self._finish(floor, out)

    def run_idle(self) -> dict:
        """wait / reject / clarify: 팔을 움직이지 않고 관찰만 한다."""
        self.stage("IDLE")
        self.move_ctrl(self.sim.data.ctrl[self.sim.arm_act].copy(), self.cfg.t_idle)
        return self._finish(None, {"fail_type": None})

    def _finish(self, floor, out):
        j = self.sim.judge(floor)
        if floor is not None and out["fail_type"] is None and not j["success"]:
            out["fail_type"] = "wrong_button" if j["other_mm"] >= 3.0 else "contact"
        if floor is None and not j["success"]:
            out["fail_type"] = "moved_when_forbidden"
        self.stage("DONE", success=j["success"], peak_mm=j["peak_mm"], other_mm=j["other_mm"])
        return {**j, **out, "t_task": round(self.sim.time - self.t0, 3), "stages": self.log}


def execute(task: dict, sim, locate=locate_button, cfg: PressConfig | None = None, status=None) -> dict:
    """작업 JSON 을 받아 실행.  task = {"intent": "press", "floor": 2, ...}"""
    ex = PressExecutor(sim, cfg, status)
    if task.get("intent") == "press" and task.get("floor") is not None:
        return ex.run_press(int(task["floor"]), locate=locate)
    return ex.run_idle()


def config_dict(cfg: PressConfig | None = None) -> dict:
    return asdict(cfg or PressConfig())
