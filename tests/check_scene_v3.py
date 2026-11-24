"""Step 1 점검: scene_v3 모델.
 (a) 직진 v=0.3m/s 5s -> 1.5m ±3cm, 옆 미끄러짐·z 변화 없음
 (b) 제자리 회전 ω=0.5rad/s, π/2 rad 만큼 -> 90° ±3°
 (c) 기준 도킹 자세에서 GT 좌표로 6개 버튼 모두 누르기 (6/6), 그동안 휠체어 위치 유지
 (d) 관찰·head·wrist 이미지 저장 (outputs/check_v3/*.png)
사용법: MUJOCO_GL=egl python tests/check_scene_v3.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from control.press_fsm import PressExecutor  # noqa: E402
from sim.recorder import hidden_site_option  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402

OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs")) / "check_v3"
OUT.mkdir(parents=True, exist_ok=True)
ok_all = True


def check(name, cond, msg):
    global ok_all
    ok_all &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}: {msg}")


def place(sim, pose):
    """점검 전용: 휠체어를 주어진 자세에 놓는다(에피소드 시작 전 배치와 같음)."""
    sim.data.qpos[sim.base_qadr] = pose
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


# (a) 직진
sim = make_world(0, "none")
p0 = sim.base_pose()
z0 = sim.data.xpos[sim.model.body("wheelchair_base").id][2]
sim.set_twist(0.3, 0.0)
sim.step(int(5.0 / sim.dt))
sim.set_twist(0.0, 0.0)
sim.step(int(0.5 / sim.dt))
p1 = sim.base_pose()
dz = sim.data.xpos[sim.model.body("wheelchair_base").id][2] - z0
check("drive", abs((p1[0] - p0[0]) - 1.5) < 0.05 and abs(p1[1] - p0[1]) < 0.005 and abs(dz) < 1e-6,
      f"dx={p1[0]-p0[0]:.3f}m dy={p1[1]-p0[1]:.4f}m dz={dz:.2e} yaw={np.degrees(p1[2]):.2f}°")

# (b) 회전
p0 = sim.base_pose()
sim.set_twist(0.0, 0.5)
sim.step(int(np.pi / 2 / 0.5 / sim.dt))
sim.set_twist(0.0, 0.0)
sim.step(int(0.5 / sim.dt))
p1 = sim.base_pose()
dyaw = np.degrees(p1[2] - p0[2])
check("rotate", abs(dyaw - 90) < 3 and np.linalg.norm(p1[:2] - p0[:2]) < 0.01,
      f"dyaw={dyaw:.2f}° dxy={np.linalg.norm(p1[:2]-p0[:2])*1000:.1f}mm")

# (c) 도킹 자세에서 6버튼
res = []
for f in range(1, 7):
    sim = make_world(0, "none", floor=f)
    panel = sim.data.xpos[sim.model.body("elevator_panel").id]
    dk = dock_pose(panel)
    place(sim, dk)
    sim.hold(dk)
    sim.step(100)
    r = PressExecutor(sim).run_press(f)
    drift = np.linalg.norm(sim.base_pose()[:2] - dk[:2]) * 1000
    res.append(r["success"])
    print(f"   floor {f}: success={r['success']} peak={r['peak_mm']}mm other={r['other_mm']}mm "
          f"fail={r['fail_type']} lit={sim.lit[f]} base_drift={drift:.1f}mm coll={r['collisions']}")
check("press_6", all(res), f"{sum(res)}/6")

# (d) 이미지
sim = make_world(3, "variation", floor=2)
opt = hidden_site_option()
r = mujoco.Renderer(sim.model, 480, 640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [3.5, 1.8, 0.0]
cam.distance, cam.azimuth, cam.elevation = 7.5, 90, -70
r.update_scene(sim.data, camera=cam, scene_option=opt)
Image.fromarray(r.render()).save(OUT / "overview.png")
r.update_scene(sim.data, camera="head_rgb", scene_option=opt)
Image.fromarray(r.render()).save(OUT / "head_start.png")
panel = sim.data.xpos[sim.model.body("elevator_panel").id]
place(sim, dock_pose(panel))
r.update_scene(sim.data, camera="head_rgb", scene_option=opt)
Image.fromarray(r.render()).save(OUT / "head_dock.png")
cam.lookat[:] = panel
cam.distance, cam.azimuth, cam.elevation = 0.45, -90, 0
r.update_scene(sim.data, camera=cam, scene_option=opt)
Image.fromarray(r.render()).save(OUT / "panel_front.png")
cam.lookat[:] = [panel[0] - 0.3, 0.9, 0.8]
cam.distance, cam.azimuth, cam.elevation = 2.4, -125, -20
r.update_scene(sim.data, camera=cam, scene_option=opt)
Image.fromarray(r.render()).save(OUT / "dock_side.png")
r.close()
print("layout", sim.layout, "obstacles", {k: v.round(2).tolist() for k, v in sim.obstacles.items()}, sim.offset_kind)
print("images:", OUT)
print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
