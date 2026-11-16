"""ACT 정책 실행부.  IK 실행부와 같은 판정 코드를 쓴다.

텍스트는 정책에 직접 들어가지 않는다: 명령 해석기가 낸 목표 층을 원-핫(3)으로 바꿔 관절 상태(7)에 붙인
observation.state(10) 로 넣는다 -> 정책 하나로 3개 층을 처리.
정책 20Hz: 행동(ctrl 7)을 한 번 내면 물리 25스텝(0.05s) 동안 유지. 고정 시간(기본 10s) 동안 실행.

LeRobot 버전마다 모듈 경로와 정규화(processor) 방식이 달라서 로드 부분을 분리해 두었다.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from policy.gen_demos import IMG_H, IMG_W, ObsSampler, onehot

RUN_SECONDS = float(os.environ.get("KWVIP_ACT_SECONDS", "10"))


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                     and torch.backends.mps.is_available() else "cpu")


class ActPolicy:
    """LeRobot ACT 체크포인트 래퍼: reset(), act(head, wrist, state) -> ctrl(7)."""

    def __init__(self, ckpt: str | Path, device: str | None = None):
        import torch
        self.torch = torch
        self.device = device or _device()
        ckpt = str(ckpt)
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
        except ImportError:  # lerobot <= 0.2
            from lerobot.common.policies.act.modeling_act import ACTPolicy
        self.policy = ACTPolicy.from_pretrained(ckpt)
        self.policy.to(self.device).eval()
        self.pre = self.post = None
        try:  # lerobot >= 0.4: 정규화가 processor 파이프라인으로 분리되어 체크포인트에 같이 저장됨
            from lerobot.policies.factory import make_pre_post_processors
            self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=ckpt)
        except Exception:
            pass
        self.ckpt = ckpt

    def reset(self):
        self.policy.reset()

    def act(self, head: np.ndarray, wrist: np.ndarray, state: np.ndarray) -> np.ndarray:
        torch = self.torch

        def img(x):
            return torch.from_numpy(x).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

        batch = {"observation.images.head": img(head), "observation.images.wrist": img(wrist),
                 "observation.state": torch.from_numpy(state).float().unsqueeze(0)}
        with torch.inference_mode():
            if self.pre is not None:
                batch = self.pre(batch)
            else:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            a = self.policy.select_action(batch)
            if self.post is not None:
                a = self.post(a)
        return a.squeeze(0).detach().cpu().numpy().astype(np.float64)


def load_policy(ckpt):
    if not ckpt:
        raise SystemExit("ACT 체크포인트 경로가 필요합니다 (--act-ckpt 또는 KWVIP_ACT_CKPT)")
    return ActPolicy(ckpt)


def execute_policy(task: dict, sim, policy: ActPolicy, status=None, seconds: float = RUN_SECONDS) -> dict:
    """고정 시간 동안 정책 실행.  반환 형식은 control.press_fsm.execute 와 같다."""
    status = status or (lambda **kw: None)
    floor = int(task["floor"])
    sampler = ObsSampler(sim, floor, IMG_H, IMG_W)  # 관측 렌더러 재사용 (hook 로 등록하지 않음)
    policy.reset()
    every = sampler.every
    t0 = sim.time
    lat = []
    status(state="ACT")
    n_steps = int(round(seconds / (every * sim.dt)))
    for _ in range(n_steps):
        head, wrist, _ = sampler.obs()
        state = np.r_[sim.joint_state7(), onehot(floor)].astype(np.float32)
        tt = time.perf_counter()
        a = policy.act(head, wrist, state)
        lat.append((time.perf_counter() - tt) * 1000)
        lo, hi = sim.model.actuator_ctrlrange[:7, 0], sim.model.actuator_ctrlrange[:7, 1]
        sim.data.ctrl[:7] = np.clip(a[:7], lo, hi)
        sim.step(every)
    sampler.close()
    j = sim.judge(floor)
    fail = None
    if not j["success"]:
        fail = "wrong_button" if j["other_mm"] >= 3 else ("contact" if j["peak_mm"] > 1 else "timeout")
    status(state="DONE")
    return {**j, "fail_type": fail, "t_task": round(sim.time - t0, 3),
            "policy_ms": round(float(np.median(lat)), 2) if lat else None,
            "policy_ms_mean": round(float(np.mean(lat)), 2) if lat else None,
            "policy_ms_max": round(float(np.max(lat)), 2) if lat else None, "stages": []}
