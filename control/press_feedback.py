"""피드백 누르기 + 판단 (scene_v3, Step 6).

상태: APPROACH(5cm 앞) -> PRESS_SERVO -> HOLD -> RETRACT -> OBSERVE -> VERIFY -> (실패 시 재인식 후 재시도, 최대 2회)

PRESS_SERVO: 접근축을 따라 2mm 씩 전진하며 매 단계 센서를 읽는다
  - 목표 버튼 변위(jointpos 센서) >= STOP_MM          -> 멈춤 (등록됨)
  - 다른 버튼 변위 >= OTHER_ABORT_MM                    -> 즉시 후퇴 (wrong_button 방지)
  - 손가락 접촉력(touch) > FORCE_MAX 인데 목표가 안 움직임 -> 후퇴 (패널을 밀고 있음)
  - 최대 깊이까지 갔는데 목표 변위 < NO_CONTACT_MM       -> 미접촉 -> 재인식 후 재시도
VERIFY: 관찰 자세에서 손목 카메라로 램프(버튼 링이 주황) 확인 + 변위 최대값 >= 10mm. 둘 다 맞아야 성공.
        불일치(센서 O, 램프 X 등)는 로그에 남긴다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.press_fsm import PressConfig, PressExecutor, _Timeout


@dataclass
class FeedbackConfig:
    approach_dist: float = 0.05
    fast_until: float = -0.012      # 앞면 기준 이 깊이까지는 한 번에 (아직 닿지 않음)
    step: float = 0.002             # 서보 전진 간격 (m)
    max_depth: float = 0.022        # 앞면 기준 최대 전진
    t_step: float = 0.12            # 한 단계 이동 시간 (s)
    stop_mm: float = 12.0
    other_abort_mm: float = 2.0
    force_max: float = 60.0         # N
    no_contact_mm: float = 2.0
    hold_s: float = 0.3
    max_attempts: int = 3
    ik_step_tol: float = 0.004


class FeedbackPresser(PressExecutor):
    def __init__(self, sim, cfg: FeedbackConfig | None = None, status=None):
        super().__init__(sim, PressConfig(timeout_s=90.0), status)
        self.fc = cfg or FeedbackConfig()
        self.touch_trace = []

    def disp_mm(self, i):
        return self.sim.button_disp(i) * 1000

    def _waypoints(self, face, into, rng):
        """접근 자세 + 앞면 기준 깊이별 관절 해 (이전 해에서 이어 풀기)."""
        fc = self.fc
        pre = face - fc.approach_dist * into
        self.ik.sync(self.sim.data)
        r = self.ik.solve(pre, into, q_seed=self.sim.arm_q(), rng=rng, n_starts=self.cfg.ik_starts)
        if not r.ok(self.cfg.ik_pos_tol, self.cfg.ik_ang_tol):
            return None, {"why": "approach_unreachable", "pos_err_mm": round(r.pos_err * 1e3, 1)}
        depths = [fc.fast_until] + list(np.arange(fc.fast_until + fc.step, fc.max_depth + 1e-9, fc.step))
        way = [(None, r.q)]
        for dep in depths:
            rs = self.ik.solve_from(face + dep * into, into, way[-1][1], iters=200)
            if rs.pos_err > fc.ik_step_tol:
                return None, {"why": "press_path_unreachable", "depth_mm": round(dep * 1e3, 1)}
            way.append((dep, rs.q))
        return way, {"pos_err_mm": round(r.pos_err * 1e3, 2), "ang_err_deg": round(r.ang_err_deg, 2)}

    def press_once(self, floor, face, into, rng) -> dict:
        """한 번 누르기 시도. 반환 {"result": pressed|no_contact|other_moved|force|ik, ...}"""
        fc, sim = self.fc, self.sim
        way, info = self._waypoints(np.asarray(face), np.asarray(into), rng)
        if way is None:
            return {"result": "ik", **info}
        others = [i for i in sim.floors if i != floor]
        self.stage("APPROACH")
        self.move_ctrl(way[0][1], 2.0)
        self.stage("PRESS_SERVO")
        res, depth_reached, fmax = "no_contact", None, 0.0
        for k, (dep, q) in enumerate(way[1:]):
            self.move_ctrl(q, 0.8 if k == 0 else fc.t_step)
            f = sim.touch()
            fmax = max(fmax, f)
            depth_reached = dep
            tgt = self.disp_mm(floor)
            oth = max(self.disp_mm(i) for i in others)
            self.touch_trace.append((round(dep * 1000, 1), round(tgt, 2), round(oth, 2), round(f, 1)))
            if oth >= fc.other_abort_mm:
                res = "other_moved"
                break
            if tgt >= fc.stop_mm:
                res = "pressed"
                self.stage("HOLD")
                self.move_ctrl(q, fc.hold_s)
                break
            if f > fc.force_max and tgt < fc.no_contact_mm:
                res = "force"
                break
        self.stage("RETRACT")
        self.move_ctrl(way[0][1], 0.8)
        return {"result": res, "depth_mm": None if depth_reached is None else round(depth_reached * 1000, 1),
                "force_max_n": round(fmax, 1), "peak_mm": round(sim.peak[floor] * 1000, 2), **info}


def press_with_feedback(sim, floor, observe, wcam, status=None, cfg: FeedbackConfig | None = None, rng=None,
                        lamp_debug=None) -> dict:
    """observe() -> {"ok", "read": {"floors": {층: {"face","into",...}}}, ...} : 관찰 자세로 가서 숫자를 다시 읽는 함수.
    반환: {success, fail_type, attempts, verify, log}"""
    from perception.digits import lamp_on

    ex = FeedbackPresser(sim, cfg, status)
    rng = rng if rng is not None else np.random.default_rng(sim.seed)
    attempts, verify = [], None
    tgt = None
    fail = None
    try:
        for a in range(ex.fc.max_attempts):
            if tgt is None:
                ob = observe()
                if not ob["ok"]:
                    fail = "ik"
                    break
                fl = ob["read"]["floors"]
                if floor not in fl:
                    fail = "digit"
                    break
                tgt = fl[floor]
            r = ex.press_once(floor, tgt["face"], tgt["into"], rng)
            attempts.append(r)
            if r["result"] == "ik":
                fail = "ik"
                break
            # VERIFY: 관찰 자세에서 램프 + 센서
            ob = observe()
            ex.stage("VERIFY")
            lit, frac = (lamp_on(wcam, sim.data, tgt["face"], -np.asarray(tgt["into"]), save=lamp_debug) if ob["ok"] else (False, 0.0))
            sensor = sim.peak[floor] * 1000 >= 10.0
            verify = {"lamp": lit, "lamp_frac": frac, "sensor_peak_mm": round(sim.peak[floor] * 1000, 2),
                      "agree": lit == sensor}
            if lit and sensor:
                fail = None
                break
            if r["result"] == "other_moved":
                fail = "wrong_button"
                break
            fail = {"no_contact": "press_no_contact", "force": "press_force"}.get(r["result"], "contact")
            # 재시도: VERIFY 때 다시 읽은 위치를 쓴다 (못 읽었으면 다음 시도에서 다시 관찰)
            tgt = ob["read"]["floors"].get(floor) if ob["ok"] else None
    except _Timeout:
        fail = "timeout"
    j = sim.judge(floor)
    if fail is None and not j["success"]:
        fail = "wrong_button" if j["other_mm"] >= 3.0 else "contact"
    ex.stage("DONE", success=j["success"] and fail is None)
    return {"success": bool(j["success"] and fail is None), "fail_type": fail, "attempts": attempts,
            "n_attempts": len(attempts), "verify": verify, "touch_trace": ex.touch_trace[-12:],
            "peak_mm": j["peak_mm"], "other_mm": j["other_mm"], "stages": ex.log}
