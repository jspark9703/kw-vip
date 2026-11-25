"""도어락 인식 점검 (scene_v4): head 카메라 도어락 확인 + 손목 카메라 키 글자 판독.
시드마다 휠체어를 정답 도킹 자세에 두고(점검 전용 배치) head 로 도어락을 찾은 뒤
팔을 관찰 자세로 움직여(물리 시뮬레이션) wrist 카메라로 12개 키를 읽는다.
지표: 도어락 원점 오차, 키별 정확도·혼동행렬(목표 >=99%), 판독 불가 수, 키 앞면 위치 오차, LED(대기) 판독.
사용법: MUJOCO_GL=egl python tests/check_keypad.py [--n 30] [--cond variation]
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
from models.make_scene_v4 import DOOR_NOMINAL_X, KEYS, LOCK_REL_NOMINAL  # noqa: E402
from perception.digits import WristCamera, led_state  # noqa: E402
from perception.panel import PanelLocator  # noqa: E402
from perception.site_gt import locate_key  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402

OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs")) / "check_keypad"
OUT.mkdir(parents=True, exist_ok=True)
PRIOR = np.array([DOOR_NOMINAL_X + LOCK_REL_NOMINAL[0], LOCK_REL_NOMINAL[1], LOCK_REL_NOMINAL[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cond", default="variation")
    ap.add_argument("--seed0", type=int, default=200)
    a = ap.parse_args()
    idx = {k: i + 1 for i, k in enumerate(KEYS)}
    conf = np.zeros((len(KEYS) + 1, len(KEYS) + 1), int)   # [정답, 판독(0=불가)]
    pos_err, lock_err, leds, fails = [], [], [], 0
    cache = {}
    for seed in range(a.seed0, a.seed0 + a.n):
        sim = make_world(seed, a.cond)
        if id(sim.model) not in cache:
            cache[id(sim.model)] = (PanelLocator(sim.model, prior=PRIOR), WristCamera(sim.model))
        ploc, wcam = cache[id(sim.model)]
        dk = dock_pose(sim.lock_pos())
        sim.data.qpos[sim.base_qadr] = dk
        mujoco.mj_forward(sim.model, sim.data)
        sim.hold(dk)
        det = ploc(sim)
        if det is None:
            print(f"seed {seed}: doorlock not found {ploc.last}", flush=True)
            fails += 1
            continue
        lock_err.append(det["err_mm"])
        r = observe_panel(sim, wcam, det["center"], det["normal"])
        if not r["ok"]:
            print(f"seed {seed}: observe failed {r}", flush=True)
            fails += 1
            continue
        read = r["read"]
        wrong = []
        for k in KEYS:
            gt = locate_key(sim, k)["face"]
            hit = [g for g, info in read["keys"].items() if np.linalg.norm(info["face"] - gt) < 0.008]
            pred = hit[0] if hit else None
            conf[idx[k], idx[pred] if pred else 0] += 1
            if pred == k:
                pos_err.append(np.linalg.norm(read["keys"][k]["face"] - gt) * 1000)
            else:
                wrong.append((k, pred))
        st, _ = led_state(wcam, sim.data, read["origin"], read["normal"], keys=read["keys"])
        leds.append(st)
        print(f"seed {seed}: lock_err {det['err_mm']:.1f}mm view {r['view']} keys {read['n_keys']} read {len(read['keys'])} "
              f"wrong {wrong} min_score {min([v['score'] for v in read['keys'].values()] or [0]):.2f} led {st}", flush=True)
        if wrong or seed == a.seed0:
            Image.fromarray(read["rect"]).save(OUT / f"rect_{seed}.png")
            Image.fromarray(read["rgb"]).save(OUT / f"wrist_{seed}.png")
    total, correct = conf[1:].sum(), sum(conf[i, i] for i in range(1, len(KEYS) + 1))
    print("\n혼동행렬 (행 정답 / 열 판독, 0=불가)")
    print("      " + " ".join(f"{c:>3s}" for c in ["-"] + KEYS))
    for k in KEYS:
        print(f"  {k:>2s}: " + " ".join(f"{v:>3d}" for v in conf[idx[k]]))
    print(f"정확도 {correct}/{total} = {100 * correct / max(total, 1):.1f}%  판독 불가 {conf[1:, 0].sum()}  관찰 실패 {fails}  "
          f"키 위치 오차 중앙값 {np.median(pos_err) if pos_err else float('nan'):.2f}mm (최대 {max(pos_err) if pos_err else float('nan'):.2f})  "
          f"도어락 원점 오차 중앙값 {np.median(lock_err) if lock_err else float('nan'):.1f}mm  LED(대기) {dict(zip(*np.unique(leds, return_counts=True)))}")
    print("images:", OUT)


if __name__ == "__main__":
    main()
