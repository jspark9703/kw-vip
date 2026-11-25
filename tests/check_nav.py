"""Step 3 점검: 주행·장애물 우회·도킹 (도킹 목표 = 시뮬레이터 좌표).
base 20 + variation 20 시드: 도달률, 충돌 0, 도킹 오차, 이동 거리, 재계획·비상정지 수, 시간.
지도 그림(점유격자 + 계획 경로 + 실제 궤적): outputs/check_nav/map_<cond>_<seed>.png
사용법: MUJOCO_GL=egl python tests/check_nav.py [--n 20] [--seeds 100 ...]
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nav.dock import dock, find_and_dock  # noqa: E402
from mission.fsm import PRIOR_LOCK  # noqa: E402
from nav.follower import Navigator, summary  # noqa: E402
from nav.grid import gt_grid  # noqa: E402
from sim.world import dock_pose, make_world  # noqa: E402

OUT = Path(os.environ.get("KWVIP_OUT", ROOT / "outputs")) / "check_nav"
OUT.mkdir(parents=True, exist_ok=True)


def draw_map(sim, nav, traj, path_png, dk):
    g = nav.grid
    gt = gt_grid(sim.model, sim.data)
    s = 4  # px per cell
    img = np.full((g.ny * s, g.nx * s, 3), 255, np.uint8)
    occ_gt = np.kron(gt.occ.T[::-1], np.ones((s, s), bool))
    img[occ_gt] = (200, 200, 200)
    occ = np.kron(g.occ.T[::-1], np.ones((s, s), bool))
    img[occ] = (40, 40, 40)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)

    def px(p):
        return ((p[0] - g.x0) / g.res * s, (g.y1 - p[1]) / g.res * s)

    for k, path in enumerate(nav.log.plans):
        col = (120, 160, 255) if k < len(nav.log.plans) - 1 else (0, 80, 255)
        dr.line([px(p) for p in path], fill=col, width=1 if k < len(nav.log.plans) - 1 else 3)
    dr.line([px(p) for p in traj], fill=(230, 60, 60), width=2)
    for p, c in ((sim.start_pose, (0, 160, 0)), (dk, (255, 140, 0))):
        x, y = px(p)
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], outline=c, width=3)
    im.save(path_png)


_LOC = {}


def run(seed, cond, save_map=False, camera=False):
    sim = make_world(seed, cond)
    traj = []
    sim.step_hooks.append(lambda s: traj.append(s.base_pose()[:2].copy()) if len(traj) == 0 or s.data.time % 0.1 < s.dt else None)
    dk = dock_pose(sim.lock_pos())
    nav = Navigator(sim)
    t = time.time()
    if camera:
        from perception.panel import PanelLocator
        key = id(sim.model)
        if key not in _LOC:
            _LOC[key] = PanelLocator(sim.model, prior=PRIOR_LOCK)
        r = find_and_dock(nav, _LOC[key], PRIOR_LOCK)   # 사전정보 = 우리 집 도어락 명목 위치만
    else:
        r = dock(nav, dk)
    j = sim.judge(None)
    out = {"seed": seed, "cond": cond, "kind": sim.offset_kind, "ok": r["ok"] and j["collisions"] == 0,
           "why": r["why"] or ("collision" if j["collisions"] else None), "side": r.get("side"),
           "dock_err_mm": round(float(np.linalg.norm(sim.base_pose()[:2] - dk[:2]) * 1000), 1) if r["ok"] else np.nan,
           "panel_err_mm": (r.get("panel_refined") or r.get("panel_first") or {}).get("err_mm"), "yaw_err_deg": round(r.get("dock_yaw_err_deg", np.nan), 2),
           "collisions": j["collisions"], "hit": j["collided_with"], **summary(nav), "wall_s": round(time.time() - t, 1)}
    if save_map:
        draw_map(sim, nav, traj, OUT / f"map_{cond}_{seed}{'_cam' if camera else ''}.png", dk)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--conds", nargs="+", default=["base", "variation"])
    ap.add_argument("--maps", type=int, default=3, help="조건별 처음 몇 개의 지도 그림 저장")
    ap.add_argument("--camera", action="store_true", help="도킹 목표를 head 카메라 도어락 인식으로")
    a = ap.parse_args()
    rows = []
    for cond in a.conds:
        seeds = a.seeds or list(range(100, 100 + a.n))
        for k, s in enumerate(seeds):
            r = run(s, cond, save_map=k < a.maps, camera=a.camera)
            rows.append(r)
            print(f"[{cond:9s}] seed {s} {r['kind']:28s} ok={r['ok']!s:5s} why={r['why']} side={r['side']} "
                  f"err={r['dock_err_mm']}mm/{r['yaw_err_deg']}° panel={r['panel_err_mm']}mm coll={r['collisions']} {r['hit']} "
                  f"replans={r['replans']} estops={r['estops']} dist={r['traveled_m']}m t={r['nav_t_s']}s ({r['wall_s']}s)",
                  flush=True)
    for cond in a.conds:
        rs = [r for r in rows if r["cond"] == cond]
        ok = [r for r in rs if r["ok"]]
        print(f"== {cond}: {len(ok)}/{len(rs)} 도달(충돌 0)  도킹 오차 중앙값 "
              f"{np.median([r['dock_err_mm'] for r in ok]) if ok else float('nan'):.1f}mm / "
              f"{np.median([r['yaw_err_deg'] for r in ok]) if ok else float('nan'):.2f}°  "
              f"시간 중앙값 {np.median([r['nav_t_s'] for r in ok]) if ok else float('nan'):.1f}s  "
              f"이동 {np.median([r['traveled_m'] for r in ok]) if ok else float('nan'):.2f}m")
    print("maps:", OUT)


if __name__ == "__main__":
    main()
