"""gen_demos.py 의 npz 에피소드 -> LeRobot 데이터셋 (로컬 root, 허브 업로드 없음).

행동 라벨: gen_demos 는 샘플 시점의 ctrl 을 저장한다. 정책은 출력을 다음 0.05s(25스텝) 동안 유지하므로
          '다음 샘플 시점의 ctrl'(한 칸 앞당김)을 action 으로 쓴다. 마지막 프레임은 그대로 반복.

사용법
  python policy/to_lerobot.py --src /results/kw-vip/datasets/press_v1_small --dst /results/kw-vip/lerobot/press_v1_small
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper", "floor1", "floor2", "floor3"]
ACTION_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def task_text(floor: int) -> str:
    return f"press elevator button for floor {floor}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--repo-id", default="kwvip/press_v1")
    ap.add_argument("--no-video", action="store_true", help="이미지를 mp4 대신 png 로 저장")
    a = ap.parse_args()
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src = Path(a.src)
    meta = json.load(open(src / "meta.json"))
    h, w = meta["image_hw"]
    eps = sorted(src.glob("episode_*.npz"))
    dst = Path(a.dst)
    if dst.exists():
        shutil.rmtree(dst)
    img = {"dtype": "image" if a.no_video else "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.head": img, "observation.images.wrist": dict(img),
        "observation.state": {"dtype": "float32", "shape": (10,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=meta["fps"], features=features, root=dst,
                               robot_type="piper_wheelchair_sim", use_videos=not a.no_video)
    add_sig = inspect.signature(ds.add_frame).parameters
    for i, p in enumerate(eps):
        z = np.load(p)
        act = np.concatenate([z["action"][1:], z["action"][-1:]], 0).astype(np.float32)
        task = task_text(int(z["floor"]))
        for t in range(len(act)):
            fr = {"observation.images.head": z["head"][t], "observation.images.wrist": z["wrist"][t],
                  "observation.state": z["state"][t].astype(np.float32), "action": act[t]}
            if "task" in add_sig:      # lerobot 0.2~0.3: add_frame(frame, task)
                ds.add_frame(fr, task=task)
            else:                      # lerobot >= 0.4: frame["task"]
                fr["task"] = task
                ds.add_frame(fr)
        ds.save_episode()
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(eps)} episodes", flush=True)
    if hasattr(ds, "finalize"):
        ds.finalize()
    print(f"saved LeRobot dataset: {dst}  ({len(eps)} episodes)")


if __name__ == "__main__":
    main()
