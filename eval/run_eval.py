"""고정 시드 평가: 기본 10회 + 변화 10회 + 명령 5회.  시도별 JSON 1줄 + mp4 저장.

사용법 (vla-shell 안, 저장소 루트에서):
  MUJOCO_GL=egl python eval/run_eval.py --runner ik_gt  --parser rules+llm
  MUJOCO_GL=egl python eval/run_eval.py --runner ik_cam --parser rules
  MUJOCO_GL=egl python eval/run_eval.py --runner act --act-ckpt outputs/train/act_press_v1/checkpoints/last/pretrained_model
  MUJOCO_GL=egl python eval/run_eval.py --scene v3 --runner ik_cam --parser rules   # 주행+숫자 버튼 (eval_config_v3.yaml)
결과: outputs/eval/<run>/trials.jsonl, summary.json, videos/*.mp4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import yaml  # noqa: E402

from eval.summarize import summarize  # noqa: E402
from pipeline import OUT, RUNNERS, close_all, run_episode  # noqa: E402
from sim.scene import make_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="기본: v2 eval_config.yaml / v3 eval_config_v3.yaml")
    ap.add_argument("--scene", default="v2", choices=["v2", "v3"])
    ap.add_argument("--runner", default="ik_gt", choices=RUNNERS)
    ap.add_argument("--parser", default="rules+llm", choices=["rules", "rules+llm"])
    ap.add_argument("--run", default=None, help="실행 이름 (기본 <runner>_<parser>)")
    ap.add_argument("--splits", nargs="+", default=["base", "variation", "commands"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--act-ckpt", default=os.environ.get("KWVIP_ACT_CKPT"))
    a = ap.parse_args()

    a.config = a.config or str(ROOT / "eval" / ("eval_config_v3.yaml" if a.scene == "v3" else "eval_config.yaml"))
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    run = a.run or f"{'v3_' if a.scene == 'v3' else ''}{a.runner}_{a.parser.replace('+', '_')}"
    out_dir = OUT / "eval" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = None
    if a.runner == "act":
        from policy.act_runner import load_policy
        policy = load_policy(a.act_ckpt)

    trials = []
    jl = open(out_dir / "trials.jsonl", "w", encoding="utf-8")

    def log(r):
        trials.append(r)
        jl.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        jl.flush()
        p = r["parsed"]
        print(f"[{r['split']:9s}] seed {r['seed']} {r['command']!r:28s} -> {p['intent']}/{p['floor']} "
              f"{'OK ' if r['success'] else 'FAIL'} tgt {r['peak_mm']}mm other {r['other_mm']}mm "
              f"{r['fail_type'] or ''} t={r['t_task_s']}s", flush=True)

    t0 = time.time()
    for split in a.splits:
        if split == "commands":
            c = cfg["commands"]
            for it in c["items"]:
                exp = it["expect"]
                vid = None if a.no_video else out_dir / "videos" / f"{run}_cmd_{it['seed']}.mp4"
                r = run_episode(it["text"], it["seed"], c["condition"], floor=exp.get("floor"), expect=exp,
                                parser=a.parser, runner=a.runner, video=vid, act_policy=policy, run_name=run, split="commands",
                                scene=a.scene)
                r["kind"] = it["kind"]
                log(r)
            continue
        sp = cfg["splits"][split]
        for k, seed in enumerate(sp["seeds"]):
            if a.scene == "v3":
                from sim.world import make_world
                floor = make_world(seed, sp["condition"]).floor
                tmpl = cfg["trial_commands"]
                text = tmpl[k % len(tmpl)].format(floor=floor)
            else:
                floor = make_scene(seed, sp["condition"]).floor  # 시드가 정한 목표 층
                text = cfg["trial_command"].format(floor=floor)
            vid = None if a.no_video else out_dir / "videos" / f"{run}_{seed}.mp4"
            r = run_episode(text, seed, sp["condition"], floor=floor, expect={"intent": "press", "floor": floor},
                            parser=a.parser, runner=a.runner, video=vid, act_policy=policy, run_name=run, split=split,
                            scene=a.scene)
            log(r)
    jl.close()
    close_all()
    s = summarize(trials)
    s.update({"run": run, "runner": a.runner, "parser": a.parser, "config": a.config, "scene": a.scene,
              "wall_total_s": round(time.time() - t0, 1)})
    json.dump(s, open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in s.items() if k != "fail_types_by_split"}, ensure_ascii=False, indent=1))
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
