"""scene_v4 점검: 모델 · 도달 범위 · 그림.
 (a) 직진 v=0.3m/s 5s -> 1.5m ±5cm, 옆 미끄러짐·z 변화 없음
 (b) 제자리 회전 90° ±3°
 (c) 도킹 자세에서 GT 좌표로 키 1·3·5·*·# 누르기 (스타일러스, 피드백) — 도어락 높이별 -> LOCK_Z_RANGE 근거
 (d) 그림 저장 (outputs/check_v4/*.png): 복도 전경, 현관문 정면, 도어락 근접, 도킹 측면, head 시점
사용법: MUJOCO_GL=egl python tests/check_scene_v4.py [--heights 1.00 1.04 ...]
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

from control.press_feedback import KeyPresser  # noqa: E402
from perception.site_gt import locate_key  # noqa: E402
from sim.recorder import hidden_site_option, observer_option  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402

OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs")) / "check_v4"
OUT.mkdir(parents=True, exist_ok=True)
ok_all = True


def check(name, cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}", flush=True)


def place(sim, pose):
    """점검 전용: 휠체어를 주어진 자세에 놓는다(에피소드 시작 전 배치와 같음)."""
    sim.data.qpos[sim.base_qadr] = pose
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heights", type=float, nargs="*", default=[1.00, 1.04, 1.08, 1.11, 1.14, 1.18, 1.22])
    ap.add_argument("--keys", default="135*#")
    a = ap.parse_args()
    # (a) 직진
    sim = make_world(0, "none")
    p0 = sim.base_pose()
    sim.set_twist(0.3, 0.0)
    sim.step(int(5.0 / sim.dt))
    sim.set_twist(0.0, 0.0)
    sim.step(int(0.5 / sim.dt))
    p1 = sim.base_pose()
    check("drive", abs((p1[0] - p0[0]) - 1.5) < 0.05 and abs(p1[1] - p0[1]) < 0.005,
          f"dx={p1[0]-p0[0]:.3f}m dy={p1[1]-p0[1]:.4f}m coll={sim.n_collision_steps}")
    # (b) 회전
    p0 = sim.base_pose()
    sim.set_twist(0.0, 0.5)
    sim.step(int(np.pi / 2 / 0.5 / sim.dt))
    sim.set_twist(0.0, 0.0)
    sim.step(int(0.5 / sim.dt))
    dyaw = np.degrees(sim.base_pose()[2] - p0[2])
    check("rotate", abs(dyaw - 90) < 3, f"dyaw={dyaw:.2f}°")
    # (c) 높이별 도달 + 누르기
    reach = {}
    for z in a.heights:
        sim = make_world(0, "none")
        sim.model.body_pos[sim.model.body("doorlock").id][2] = z
        mujoco.mj_forward(sim.model, sim.data)
        dk = dock_pose(sim.lock_pos())
        place(sim, dk)
        sim.hold(dk)
        sim.step(100)
        ex = KeyPresser(sim)
        rng = np.random.default_rng(0)
        res = []
        for i, k in enumerate(a.keys):
            t = locate_key(sim, k)
            r = ex.press_key(k, t["face"], t["into"], rng, first=i == 0)
            res.append(r["result"])
        ok = sim.entered() == a.keys
        reach[z] = ok
        print(f"   lock z={z:.2f}: entered={sim.entered()!r} results={res} peaks="
              f"{ {k: round(sim.key_peak[k]*1000, 1) for k in a.keys} } coll={sim.n_collision_steps} "
              f"Fmax={max([p[4] for p in ex.trace] or [0])}N", flush=True)
    good = [z for z, ok in reach.items() if ok]
    check("reach", len(good) >= 3, f"키 {a.keys} 모두 입력된 높이: {good}")
    # (d) 그림
    sim = make_world(3, "variation")
    opt = observer_option()
    r = mujoco.Renderer(sim.model, 600, 960)
    cam = mujoco.MjvCamera()
    lock = sim.lock_pos()

    def shot(name, lookat, dist, az, el, camera=None):
        if camera:
            r.update_scene(sim.data, camera=camera, scene_option=hidden_site_option())
        else:
            cam.lookat[:] = lookat
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(sim.data, camera=cam, scene_option=opt)
        Image.fromarray(r.render()).save(OUT / f"{name}.png")
    shot("overview", [3.5, 1.6, 0.0], 7.8, -90, -62)
    shot("corridor_view", [4.6, 1.0, 1.0], 3.6, -15, -8)
    shot("door_front", [sim.door_x, 0.0, 1.15], 2.6, -90, -5)
    shot("doorlock_close", lock + [0, 0, 0.0], 0.32, -90, 0)
    shot("head_start", None, 0, 0, 0, camera="head_rgb")
    place(sim, dock_pose(lock))
    shot("head_dock", None, 0, 0, 0, camera="head_rgb")
    shot("dock_side", [lock[0] - 0.3, 0.8, 0.8], 2.6, -125, -20)
    r.close()
    print("password", sim.password, "obstacles", {k: v.round(2).tolist() for k, v in sim.obstacles.items()}, sim.offset_kind)
    print("images:", OUT)
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
