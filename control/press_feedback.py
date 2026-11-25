"""도어락 비밀번호 입력: 키마다 피드백 누르기 + 판단 (scene_v4).

순서: (관찰 자세에서 키 글자 판독) -> 키마다 APPROACH(3cm 앞) -> PRESS_SERVO -> HOLD -> RELEASE(3cm 앞으로) -> 다음 키
      -> '#' -> 관찰 자세 -> VERIFY(상태 LED)

PRESS_SERVO: 접근축을 따라 1mm 씩 전진하며 매 단계 센서를 읽는다
  - 목표 키 변위(jointpos) >= STOP_MM                      -> 멈춤 (도어락이 2.5mm 에서 입력으로 받음)
  - 다른 키 변위 >= OTHER_ABORT_MM                          -> 즉시 후퇴 (옆 키 입력 방지)
  - 스타일러스 접촉력(touch) > FORCE_MAX 인데 목표가 안 움직임 -> 후퇴 (하우징을 밀고 있음)
  - 최대 깊이까지 갔는데 목표 변위 < NO_CONTACT_MM          -> 미접촉 -> 키패드 재판독 후 그 키만 재시도 (최대 2회)
  - 그 키 목표에 IK 가 닿지 않음                           -> 재판독 후 재시도 (위치 추정이 틀렸을 수 있음)
입력 확인: 키를 뗀 뒤 '키 센서로 본 입력 기록'의 마지막이 그 키인지 확인. 다른 키가 입력되면 '*'(지움)를 누르고
           처음부터 한 번 다시 입력한다.
VERIFY: '#' 뒤 손목 카메라로 상태 LED 가 초록(열림)인지 + 키 센서 입력 기록이 비밀번호+'#' 인지. 둘 다 맞아야 성공.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from control.press_fsm import PressConfig, PressExecutor, _Timeout
from models.make_scene_v4 import LED_Z

LED_AIM = LED_Z - 0.010   # 관찰 시선 중심을 키패드 중심(원점 +1cm) -> LED 높이로


@dataclass
class KeyPressConfig:
    approach_dist: float = 0.030
    fast_until: float = -0.004      # 앞면 기준 이 깊이까지는 한 번에 (아직 닿지 않음)
    step: float = 0.001             # 서보 전진 간격 (m)
    max_depth: float = 0.012        # 앞면 기준 최대 '명령' 전진: 팔 위치 서보 정상오차(~3mm) + 접촉 침투(~1.5mm) 만큼 더 민다.
                                    # 키는 4mm 에서 바닥에 닿고, 그 뒤로는 힘이 커져 force_max 에서 멈춘다
    t_step: float = 0.12
    stop_mm: float = 3.0
    other_abort_mm: float = 1.5
    force_max: float = 40.0         # N
    no_contact_mm: float = 1.0
    hold_s: float = 0.12
    key_attempts: int = 2
    restarts: int = 1
    ik_step_tol: float = 0.002


class KeyPresser(PressExecutor):
    def __init__(self, sim, cfg: KeyPressConfig | None = None, status=None):
        super().__init__(sim, PressConfig(timeout_s=150.0), status)
        self.kc = cfg or KeyPressConfig()
        self.trace = []

    def _waypoints(self, face, into, rng):
        kc = self.kc
        pre = face - kc.approach_dist * into
        self.ik.sync(self.sim.data)
        depths = [kc.fast_until] + list(np.arange(kc.fast_until + kc.step, kc.max_depth + 1e-9, kc.step))
        info = {"why": "approach_unreachable"}
        # 접근 자세 해가 관절 한계 근처면 안쪽 경로가 끊길 수 있다 -> 다른 접근 해(시드)로 최대 4번 다시
        for attempt in range(4):
            seed = self.sim.arm_q() if attempt == 0 else None
            r = self.ik.solve(pre, into, q_seed=seed, rng=rng, n_starts=self.cfg.ik_starts)
            if not r.ok(self.cfg.ik_pos_tol, self.cfg.ik_ang_tol):
                info = {"why": "approach_unreachable", "pos_err_mm": round(r.pos_err * 1e3, 1)}
                continue
            way = [(None, r.q)]
            for dep in depths:
                rs = self.ik.solve_from(face + dep * into, into, way[-1][1], iters=400)
                if rs.pos_err > kc.ik_step_tol:
                    info = {"why": "press_path_unreachable", "depth_mm": round(dep * 1e3, 1)}
                    break
                way.append((dep, rs.q))
            else:
                return way, {"pos_err_mm": round(r.pos_err * 1e3, 2), "ik_attempt": attempt}
        return None, info

    def press_key(self, key, face, into, rng, first=False) -> dict:
        """한 키 누르기. 반환 {"result": pressed|no_contact|other_moved|force|ik, ...}"""
        kc, sim = self.kc, self.sim
        way, info = self._waypoints(np.asarray(face), np.asarray(into), rng)
        if way is None:
            return {"key": key, "result": "ik", **info}
        others = [k for k in sim.keys if k != key]
        self.stage(f"APPROACH_{key}")
        self.move_ctrl(way[0][1], 2.0 if first else 0.9)
        self.stage(f"PRESS_{key}")
        res, fmax, depth = "no_contact", 0.0, None
        for i, (dep, q) in enumerate(way[1:]):
            self.move_ctrl(q, 0.35 if i == 0 else kc.t_step)
            f = sim.touch()
            fmax = max(fmax, f)
            depth = dep
            tgt = sim.key_disp(key) * 1000
            oth = max(sim.key_disp(k) for k in others) * 1000
            self.trace.append((key, round(dep * 1000, 1), round(tgt, 2), round(oth, 2), round(f, 1)))
            if oth >= kc.other_abort_mm:
                res = "other_moved"
                break
            if tgt >= kc.stop_mm:
                res = "pressed"
                self.move_ctrl(q, kc.hold_s)
                break
            if f > kc.force_max and tgt < kc.no_contact_mm:
                res = "force"
                break
        self.move_ctrl(way[0][1], 0.5)      # 떼기: 접근 자세로
        sim.step(int(0.1 / sim.dt))
        return {"key": key, "result": res, "depth_mm": None if depth is None else round(depth * 1000, 1),
                "force_max_n": round(fmax, 1), "peak_mm": round(sim.key_peak[key] * 1000, 2), **info}


def enter_password(sim, password, observe, wcam, status=None, cfg: KeyPressConfig | None = None, rng=None,
                   first_read=None, led_debug=None) -> dict:
    """observe(aim=0.0) -> {"ok", "read": {"keys": {키: {"face","into",...}}, "origin", "normal"}} : 관찰 자세에서 다시 판독.
    반환: {success, fail_type, presses, n_presses, restarts, verify, entered}"""
    from perception.digits import led_state

    ex = KeyPresser(sim, cfg, status)
    rng = rng if rng is not None else np.random.default_rng(sim.seed)
    seq = list(password) + ["#"]
    presses, verify, fail = [], None, None
    restarts = 0
    try:
        ob = first_read or observe()
        if not ob["ok"]:
            return _done(ex, sim, "ik", presses, restarts, verify)
        targets = ob["read"]["keys"]
        missing = sorted(set(seq) - set(targets))
        if missing:
            return _done(ex, sim, "digit", presses, restarts, verify, missing=missing)
        i = 0
        while i < len(seq):
            k = seq[i]
            n_before = len(sim.events)
            ok = False
            for att in range(ex.kc.key_attempts):
                r = ex.press_key(k, targets[k]["face"], targets[k]["into"], rng, first=not presses)
                presses.append(r)
                new = [e for e, _ in sim.events[n_before:]]
                if r["result"] == "ik" and att == ex.kc.key_attempts - 1:
                    return _done(ex, sim, "ik", presses, restarts, verify)
                if new and new != [k]:                 # 다른 키가 입력됨 -> '*' 로 지우고 처음부터
                    break
                if new == [k]:
                    ok = True
                    break
                ob = observe()                          # 미접촉 -> 다시 판독 후 그 키만 재시도
                if ob["ok"] and k in ob["read"]["keys"]:
                    targets = {**targets, **ob["read"]["keys"]}
            if ok:
                i += 1
                continue
            new = [e for e, _ in sim.events[n_before:]]
            if new and new != [k] and restarts < ex.kc.restarts and "*" in targets:
                restarts += 1
                ex.stage("CLEAR")
                ex.press_key("*", targets["*"]["face"], targets["*"]["into"], rng)
                i = 0
                continue
            fail = "wrong_key" if new else "press_no_contact"
            break
        # VERIFY: 관찰 자세에서 상태 LED
        ob2 = observe()
        ex.stage("VERIFY")
        if ob2["ok"]:
            st, frac = led_state(wcam, sim.data, ob2["read"]["origin"], ob2["read"]["normal"], save=led_debug,
                                 keys=ob2["read"]["keys"])
        else:
            st, frac = "unseen", 0.0
        rechecked = False
        if st != "open":       # LED 가 화면 가장자리라 작게 보였을 수 있다 -> LED 를 가운데에 두는 시점에서 한 번 더
            ob3 = observe(aim=LED_AIM)
            if ob3["ok"]:
                rechecked = True
                st, frac = led_state(wcam, sim.data, ob3["read"]["origin"], ob3["read"]["normal"], save=led_debug,
                                     keys=ob3["read"]["keys"] or ob2["read"]["keys"])
        sensor_ok = sim.entered().split("*")[-1] == "".join(seq)   # 키 센서로 본 입력 기록
        verify = {"led": st, "led_frac": frac, "led_rechecked": rechecked, "sensor_seq_ok": bool(sensor_ok),
                  "agree": (st == "open") == bool(sensor_ok)}
        if fail is None and not (st == "open" and sensor_ok):
            fail = "unlock_failed"
    except _Timeout:
        fail = "timeout"
    return _done(ex, sim, fail, presses, restarts, verify)


def _done(ex, sim, fail, presses, restarts, verify, **kw):
    ex.stage("DONE")
    return {"fail_type": fail, "presses": presses, "n_presses": len(presses), "restarts": restarts,
            "verify": verify, "entered": sim.entered(), "trace": ex.trace[-16:], "stages": ex.log, **kw}
