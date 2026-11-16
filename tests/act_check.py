"""ACT 체크포인트 점검 (평가 시드 사용 금지): 개발 시드에서 성공률·추론 지연.
사용법: python tests/act_check.py <ckpt> [--seeds 0 10] [--condition train] [--video-first]
"""
import argparse, os, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")
from pipeline import OUT, run_episode
from policy.act_runner import load_policy
from sim.scene import make_scene
ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--seeds", nargs=2, type=int, default=[0, 10])
ap.add_argument("--condition", default="train"); ap.add_argument("--video-first", action="store_true")
a = ap.parse_args()
assert a.seeds[1] <= 10000, "평가 시드 사용 금지"
pol = load_policy(a.ckpt); print("device", pol.device)
ok, lat = 0, []
for i, seed in enumerate(range(*a.seeds)):
    f = make_scene(seed, a.condition).floor
    vid = OUT / "videos" / f"act_check_{seed}.mp4" if (a.video_first and i == 0) else None
    r = run_episode(f"{f}층 눌러줘", seed, a.condition, floor=f, expect={"intent": "press", "floor": f},
                    parser="rules", runner="act", act_policy=pol, video=vid)
    ok += r["success"]; lat.append(r["policy_ms"])
    print(seed, f, "OK " if r["success"] else "FAIL", r["peaks_mm"], r["fail_type"], f"{r['policy_ms']}ms")
n = a.seeds[1] - a.seeds[0]
print(f"ACT {ok}/{n} success, policy_ms median {statistics.median(lat):.2f}")
