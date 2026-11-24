"""Step 5 점검: 숫자 인식 정확도.
시드마다 휠체어를 정답 도킹 자세에 두고(점검 전용 배치), head 카메라로 패널을 찾은 뒤
팔을 관찰 자세로 움직여(물리 시뮬레이션) wrist 카메라로 숫자를 읽는다.
지표: 버튼별 정확도·혼동행렬(목표 >=99%), 판독 불가 수, 버튼 앞면 위치 오차(GT 대비) 중앙값.
사용법: MUJOCO_GL=egl python tests/check_digits.py [--n 50] [--cond variation]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from control.observe import observe_panel  # noqa: E402
from perception.digits import WristCamera  # noqa: E402
from perception.panel import PanelLocator  # noqa: E402
from perception.site_gt import locate_button  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402

OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs")) / "check_digits"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cond", default="variation")
    ap.add_argument("--seed0", type=int, default=200)
    a = ap.parse_args()
    conf = np.zeros((7, 10), int)   # [정답 층, 판독 층(0=판독 불가)]
    pos_err, n_fail_obs = [], 0
    cache = {}
    for seed in range(a.seed0, a.seed0 + a.n):
        sim = make_world(seed, a.cond)
        key = id(sim.model)
        if key not in cache:
            cache[key] = (PanelLocator(sim.model), WristCamera(sim.model))
        ploc, wcam = cache[key]
        panel = sim.data.xpos[sim.model.body("elevator_panel").id].copy()
        sim.data.qpos[sim.base_qadr] = dock_pose(panel)
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold()
        det = ploc(sim)
        if det is None:
            print(f"seed {seed}: panel not found")
            n_fail_obs += 1
            continue
        r = observe_panel(sim, wcam, det["center"], det["normal"])
        if not r["ok"]:
            print(f"seed {seed}: observe failed {r['why']}")
            n_fail_obs += 1
            continue
        read = r["read"]
        got = {v: k for k, v in {f: read["floors"][f]["px"] for f in read["floors"]}.items()}
        wrong = []
        for f in sim.floors:
            gt = locate_button(sim, f)["face"]
            # 어느 판독 결과가 이 버튼(위치)에 해당하는가
            hit = [g for g, info in read["floors"].items() if np.linalg.norm(info["face"] - gt) < 0.02]
            pred = hit[0] if hit else 0
            conf[f, pred] += 1
            if pred == f:
                pos_err.append(np.linalg.norm(read["floors"][f]["face"] - gt) * 1000)
            else:
                wrong.append((f, pred))
        print(f"seed {seed}: layout {sim.layout} buttons {read['n_buttons']} read {sorted(read['floors'])} "
              f"wrong {wrong} min_score {min([v['score'] for v in read['floors'].values()] or [0]):.2f}", flush=True)
        if wrong or seed == a.seed0:
            Image.fromarray(read["rect"]).save(OUT / f"rect_{seed}.png")
            Image.fromarray(read["rgb"]).save(OUT / f"wrist_{seed}.png")
    total = conf[1:].sum()
    correct = sum(conf[f, f] for f in range(1, 7))
    print("\n혼동행렬 (행 정답 1..6 / 열 판독 0=불가,1..9)")
    print("     " + " ".join(f"{j:>3d}" for j in range(10)))
    for f in range(1, 7):
        print(f"  {f}: " + " ".join(f"{v:>3d}" for v in conf[f]))
    print(f"정확도 {correct}/{total} = {100 * correct / max(total, 1):.1f}%  판독 불가 {conf[1:, 0].sum()}  "
          f"관찰 실패 {n_fail_obs}  버튼 위치 오차 중앙값 {np.median(pos_err) if pos_err else float('nan'):.2f}mm "
          f"(최대 {max(pos_err) if pos_err else float('nan'):.2f}mm)")
    print("images:", OUT)


if __name__ == "__main__":
    main()
