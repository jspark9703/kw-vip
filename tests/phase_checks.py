"""단계별 통과 조건 점검.
  phase0 : 기본 위치에서 3개 층 모두 10mm 이상 눌림 + mp4 녹화
  phase1 : 학습 범위(train) 무작위 30회 중 28회 이상 성공 (개발 시드 0~29)
  reach  : 변화 조건 범위가 도달 가능한지 사전 확인 (평가와 겹치지 않는 시드 40000~40049)
사용법: MUJOCO_GL=egl python tests/phase_checks.py [phase0 phase1 reach]
"""
import os, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
from control.press_fsm import execute
from pipeline import OUT, run_episode
from sim.scene import make_scene

which = sys.argv[1:] or ["phase0", "phase1", "reach"]
ok_all = True
if "phase0" in which:
    res = []
    for f in (1, 2, 3):
        r = run_episode(f"{f}층 눌러줘", 0, "none", floor=f, parser="rules",
                        video=OUT / "videos" / f"phase0_floor{f}.mp4", run_name="phase0")
        res.append(r)
        print(f"phase0 floor {f}: {r['peak_mm']}mm other {r['other_mm']}mm video={r['video']}")
    p = all(r["success"] for r in res) and all(Path(ROOT / r["video"]).stat().st_size > 10000 for r in res)
    print("PHASE0", "PASS" if p else "FAIL"); ok_all &= p
if "phase1" in which:
    n_ok, fails = 0, Counter()
    for seed in range(30):
        sim = make_scene(seed, "train")
        r = execute({"intent": "press", "floor": sim.floor}, sim)
        n_ok += r["success"]; fails[r["fail_type"]] += not r["success"]
    p = n_ok >= 28
    print(f"PHASE1 {n_ok}/30 success, fails={dict(+fails)} ->", "PASS" if p else "FAIL"); ok_all &= p
if "reach" in which:
    n_ok, fails, kinds = 0, Counter(), Counter()
    for seed in range(40000, 40050):
        sim = make_scene(seed, "variation")
        r = execute({"intent": "press", "floor": sim.floor}, sim)
        n_ok += r["success"]; kinds[sim.offset_kind] += 1
        if not r["success"]:
            fails[r["fail_type"]] += 1
            print("  fail", seed, sim.floor, sim.offset_kind, sim.panel_offset.round(3), r["fail_type"], r["ik"])
    print(f"REACH(variation, IK+GT) {n_ok}/50 kinds={dict(kinds)} fails={dict(fails)}")
sys.exit(0 if ok_all else 1)
