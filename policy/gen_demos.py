"""ACT 학습용 데모 생성: Phase 1 의 IK 상태 기계를 전문가로 사용.

관측 (20Hz)
  observation.images.head  : head_rgb  (H x W x 3, uint8)
  observation.images.wrist : wrist_rgb (H x W x 3, uint8)
  observation.state        : 관절 7 (joint1~6 + 그리퍼) + 목표 층 원-핫 3  = 10차원
행동
  action                   : ctrl 7 (관절 목표각 6 + 그리퍼)   -> 실행 시 그대로 data.ctrl 에 넣음
정책 20Hz = 물리 500Hz 에서 25스텝마다 1회.

데이터 다양화: 일부 에피소드에서 접근 직전 관절에 작은 잡음(ctrl 교란)을 넣고, IK 가 다시 계획해 복구하는 궤적을 포함.
실패한 데모는 저장하지 않는다 (meta.json 에 개수 기록).

사용법
  MUJOCO_GL=egl python policy/gen_demos.py --per-floor 10 --out outputs/datasets/press_v1_small   # 형식 확인용
  MUJOCO_GL=egl python policy/gen_demos.py --per-floor 80 --out outputs/datasets/press_v1
출력: episode_000000.npz ... + meta.json   (LeRobot 변환은 policy/to_lerobot.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from control.press_fsm import PressExecutor  # noqa: E402
from sim.recorder import hidden_site_option  # noqa: E402
from sim.scene import make_scene  # noqa: E402

FPS = 20
IMG_H, IMG_W = 120, 160  # 작게 시작 (ACT backbone 입력은 학습 설정에서 리사이즈 가능)


def onehot(floor: int) -> np.ndarray:
    v = np.zeros(3, np.float32)
    v[floor - 1] = 1
    return v


class ObsSampler:
    """sim.step_hooks 에 등록: 25스텝마다 (이미지, 상태, 행동=현재 ctrl) 저장."""

    def __init__(self, sim, floor, h=IMG_H, w=IMG_W):
        self.sim, self.floor = sim, floor
        self.r = mujoco.Renderer(sim.model, h, w)
        self.opt = hidden_site_option()
        self.every = int(round(1.0 / (FPS * sim.dt)))
        self.k = 0
        self.buf = {"head": [], "wrist": [], "state": [], "action": [], "t": []}

    def obs(self):
        d = self.sim.data
        self.r.update_scene(d, camera="head_rgb", scene_option=self.opt)
        head = self.r.render().copy()
        self.r.update_scene(d, camera="wrist_rgb", scene_option=self.opt)
        wrist = self.r.render().copy()
        state = np.r_[self.sim.joint_state7(), onehot(self.floor)].astype(np.float32)
        return head, wrist, state

    def __call__(self, sim):
        if self.k % self.every == 0:
            head, wrist, state = self.obs()
            b = self.buf
            b["head"].append(head)
            b["wrist"].append(wrist)
            b["state"].append(state)
            b["action"].append(sim.data.ctrl[:7].astype(np.float32).copy())
            b["t"].append(sim.time)
        self.k += 1

    def close(self):
        self.r.close()


def run_demo_episode(seed, condition, floor, noise=0.0, hold_s=1.0):
    sim = make_scene(seed, condition, floor=floor)
    sampler = ObsSampler(sim, floor)
    sim.step_hooks.append(sampler)
    ex = PressExecutor(sim)
    rng = np.random.default_rng(seed + 7)
    if noise > 0:  # 교란: 시작 직후 관절 목표에 잡음 -> 관측 상태가 다양해지고, 이후 IK 가 현재 자세에서 재계획
        q = sim.data.ctrl[:6] + rng.normal(0, noise, 6)
        ex.move_ctrl(np.clip(q, ex.ik.lo, ex.ik.hi), 0.8)
    res = ex.run_press(floor, rng=rng)
    ex.move_ctrl(sim.data.ctrl[:6].copy(), hold_s)  # 끝난 뒤 정지 구간 (정책이 '멈춤'도 배우도록)
    sampler.close()
    b = {k: np.asarray(v) for k, v in sampler.buf.items()}
    return res, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-floor", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=999, help="개발 시드 범위 끝 (평가 시드 10000+ 사용 금지)")
    ap.add_argument("--noise-frac", type=float, default=0.3, help="교란 에피소드 비율")
    ap.add_argument("--noise-rad", type=float, default=0.15)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "datasets" / "press_v1"))
    a = ap.parse_args()
    assert a.seed_end < 10000, "평가 시드(10000~)와 겹치면 안 됨"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    need = {1: a.per_floor, 2: a.per_floor, 3: a.per_floor}
    ep, n_fail, seed, t0 = 0, 0, a.seed_start, time.time()
    rng = np.random.default_rng(12345)
    while any(v > 0 for v in need.values()):
        if seed > a.seed_end:
            print("개발 시드 범위를 다 썼습니다")
            break
        sim0 = make_scene(seed, "train")
        floor = sim0.floor
        if need[floor] <= 0:  # 층 균형을 위해 남은 층으로 바꿈 (같은 시드의 패널 위치는 유지)
            floor = max(need, key=need.get)
        noise = a.noise_rad if rng.random() < a.noise_frac else 0.0
        res, b = run_demo_episode(seed, "train", floor, noise)
        if not res["success"]:
            n_fail += 1
            print(f"  skip seed {seed} floor {floor}: {res['fail_type']}")
        else:
            np.savez_compressed(out / f"episode_{ep:06d}.npz", seed=seed, floor=floor, noise=noise, **b)
            need[floor] -= 1
            ep += 1
            if ep % 10 == 0:
                print(f"  {ep} episodes ({time.time() - t0:.0f}s) last len={len(b['action'])}", flush=True)
        seed += 1
    meta = {"fps": FPS, "image_hw": [IMG_H, IMG_W], "n_episodes": ep, "n_failed_skipped": n_fail,
            "seeds_used": [a.seed_start, seed - 1], "condition": "train (x±2cm, z±2cm)",
            "per_floor": a.per_floor, "noise_frac": a.noise_frac, "noise_rad": a.noise_rad,
            "state": "joint1..6, joint7(gripper), floor_onehot(3)", "action": "ctrl[0:7]",
            "expert": "IK press FSM (control/press_fsm.py)", "gen_time_s": round(time.time() - t0, 1)}
    json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
