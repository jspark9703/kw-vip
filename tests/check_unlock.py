"""비밀번호 입력 점검 (scene_v4): 정답 도킹 자세에서 head 도어락 인식 -> 키 판독 -> 등록 비밀번호 + '#' 입력 -> LED 판단.
지표: 성공률, 키당 누름 수, '*' 재입력, LED·센서 일치, 접촉력, 시간.
--inject-miss d : 첫 키 목표를 도어락 가로 방향으로 d(m) 틀리게 (키 사이 틈을 겨눔) -> 재판독·재시도 경로 점검
사용법: MUJOCO_GL=egl python tests/check_unlock.py [--n 10] [--cond variation] [--inject-miss 0.010]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from control.observe import observe_panel  # noqa: E402
from control.press_feedback import enter_password  # noqa: E402
from mission.fsm import PRIOR_LOCK  # noqa: E402
from perception.digits import WristCamera  # noqa: E402
from perception.panel import PanelLocator  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cond", default="variation")
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--inject-miss", type=float, default=0.0)
    a = ap.parse_args()
    cache, rows = {}, []
    for seed in range(a.seed0, a.seed0 + a.n):
        sim = make_world(seed, a.cond)
        if id(sim.model) not in cache:
            cache[id(sim.model)] = (PanelLocator(sim.model, prior=PRIOR_LOCK), WristCamera(sim.model))
        ploc, wcam = cache[id(sim.model)]
        dk = dock_pose(sim.lock_pos())
        sim.data.qpos[sim.base_qadr] = dk
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold(dk)
        det = ploc(sim)
        calls = {"n": 0}

        def observe(aim=0.0):
            ob = observe_panel(sim, wcam, det["center"], det["normal"], aim=aim)
            calls["n"] += 1
            if calls["n"] == 1 and a.inject_miss and ob["ok"]:
                k0 = sim.password[0]
                t = dict(ob["read"]["keys"][k0])
                hz = np.cross([0, 0, 1.0], -np.asarray(t["into"]))
                t["face"] = t["face"] + a.inject_miss * hz
                ob["read"]["keys"][k0] = t
            return ob

        t0 = time.time()
        r = enter_password(sim, sim.password, observe, wcam)
        j = sim.judge(sim.password)
        rows.append((j, r))
        v = r["verify"] or {}
        print(f"seed {seed} pw {len(sim.password)}자리: success={j['success']} fail={r['fail_type']} presses={r['n_presses']} "
              f"restarts={r['restarts']} results={[p['result'] for p in r['presses']]} wrong_inputs={j.get('wrong_inputs')} "
              f"led={v.get('led')} agree={v.get('agree')} Fmax={max([p.get('force_max_n', 0) for p in r['presses']] or [0])}N "
              f"t_sim={sim.time:.1f}s ({time.time()-t0:.1f}s)", flush=True)
    ok = sum(j["success"] for j, _ in rows)
    keys = sum(len(j["target"]) for j, _ in rows)
    print(f"== 성공 {ok}/{len(rows)}  키당 누름 {sum(r['n_presses'] for _, r in rows) / max(keys, 1):.3f}  "
          f"'*' 재입력 {sum(1 for _, r in rows if r['restarts'])}  LED=센서 {sum(bool((r['verify'] or {}).get('agree')) for _, r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
