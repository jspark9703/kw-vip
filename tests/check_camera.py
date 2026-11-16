"""카메라 인식 위치 오차 점검 (개발용 시드).  사용법: MUJOCO_GL=egl python tests/check_camera.py"""
import sys, time; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.scene import make_scene
from perception.camera import CameraLocator
loc = None
for cond in ("none", "train", "variation"):
    for seed in range(5):
        sim = make_scene(500 + seed, cond)
        loc = loc or CameraLocator(sim.model)
        t = time.perf_counter()
        errs = []
        for f in (1, 2, 3):
            r = loc(sim, f)
            errs.append(f"{f}:{r.get('err_mm', 'X')}mm/{r.get('normal_err_deg', '-')}deg" if r.get("face") is not None else f"{f}:FAIL {r['info']}")
        print(cond, sim.offset_kind, sim.panel_offset.round(3), " ".join(errs), f"{(time.perf_counter()-t)*1000/3:.0f}ms/det")
loc.close()
