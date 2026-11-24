"""Step 6 점검: 피드백 누르기 + 판단.
정답 도킹 자세에 휠체어를 두고(점검 전용 배치) head 패널 인식 -> 관찰 자세 숫자 판독 -> 피드백 누르기 -> VERIFY.
지표: 성공률, 시도 수, 램프·센서 일치, 누를 때 최대 접촉력, 시간.
사용법: MUJOCO_GL=egl python tests/check_press_v3.py [--n 12] [--cond variation]
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
from control.press_feedback import press_with_feedback  # noqa: E402
from perception.digits import WristCamera  # noqa: E402
from perception.panel import PanelLocator  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cond", default="variation")
    ap.add_argument("--seed0", type=int, default=300)
    ap.add_argument("--dock-err", type=float, default=0.0, help="도킹 오차 모의 (m, x 방향)")
    ap.add_argument("--inject-miss", type=float, default=0.0,
                    help="첫 시도 목표를 패널 가로 방향으로 이만큼(m) 틀리게 (재시도 경로 점검)")
    a = ap.parse_args()
    cache, rows = {}, []
    for seed in range(a.seed0, a.seed0 + a.n):
        sim = make_world(seed, a.cond)
        if id(sim.model) not in cache:
            cache[id(sim.model)] = (PanelLocator(sim.model), WristCamera(sim.model))
        ploc, wcam = cache[id(sim.model)]
        panel = sim.data.xpos[sim.model.body("elevator_panel").id].copy()
        dk = dock_pose(panel) + np.array([a.dock_err, 0, 0])
        sim.data.qpos[sim.base_qadr] = dk
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold(dk)
        det = ploc(sim)
        t0 = time.time()
        dbg = ROOT / "outputs" / "tmp" / f"lamp_{seed}.png"
        calls = {"n": 0}

        def observe():
            ob = observe_panel(sim, wcam, det["center"], det["normal"])
            calls["n"] += 1
            if calls["n"] == 1 and a.inject_miss and ob["ok"] and sim.floor in ob["read"]["floors"]:
                t = dict(ob["read"]["floors"][sim.floor])
                hz = np.cross([0, 0, 1.0], -np.asarray(t["into"]))
                t["face"] = t["face"] + a.inject_miss * hz       # 버튼 사이 빈 곳을 겨누게
                ob["read"]["floors"][sim.floor] = t
            return ob

        r = press_with_feedback(sim, sim.floor, observe, wcam, lamp_debug=dbg)
        # 램프 오검출 점검: 누르지 않은 다른 버튼 하나
        from perception.digits import lamp_on
        ob = observe_panel(sim, wcam, det["center"], det["normal"])
        other = [f for f in ob["read"]["floors"] if f != sim.floor][:1] if ob["ok"] else []
        r["lamp_other"] = lamp_on(wcam, sim.data, ob["read"]["floors"][other[0]]["face"], det["normal"])[0] if other else None
        rows.append(r)
        v = r["verify"] or {}
        print(f"seed {seed} floor {sim.floor}: success={r['success']} fail={r['fail_type']} attempts={r['n_attempts']} "
              f"results={[x['result'] for x in r['attempts']]} peak={r['peak_mm']} other={r['other_mm']} "
              f"lamp={v.get('lamp')}({v.get('lamp_frac')}) agree={v.get('agree')} lamp_other={r['lamp_other']} "
              f"Fmax={max([x.get('force_max_n', 0) for x in r['attempts']] or [0])}N t_sim={sim.time:.1f}s ({time.time()-t0:.1f}s)",
              flush=True)
    ok = sum(r["success"] for r in rows)
    print(f"== 성공 {ok}/{len(rows)}  평균 시도 {np.mean([r['n_attempts'] for r in rows]):.2f}  "
          f"램프·센서 일치 {sum(bool((r['verify'] or {}).get('agree')) for r in rows)}/{len(rows)}  "
          f"다른 버튼 램프 오검출 {sum(bool(r['lamp_other']) for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
