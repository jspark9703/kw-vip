"""press_tip 기준 DLS(감쇠 최소제곱) IK.

과업 공간 오차 = [위치 3 | 방향 3].  방향 항은 link6 의 z 축(손가락 방향)을 누르는 방향 `zdir` 에
맞추는 cross-product 오차이므로 실제 구속은 5자유도(위치 3 + 접근축 2)이고 joint6 회전은 자유다.
(문서의 '6D DLS'는 6행 야코비안을 쓴다는 뜻이며, 접근축 주위 회전은 구속하지 않는다.)

- IK 는 별도 MjData 복사본에서만 계산 -> 실제 시뮬레이션 상태를 건드리지 않음
- 다중 초기값 + 이전 해에서 이어 풀기(warm start) -> 손목이 갑자기 뒤집히는 해를 피함
- 팔 관절(joint1~6)의 qpos/dof 주소는 이름으로 찾는다 -> 주행 베이스 관절이 앞에 붙은 scene_v3 에서도 동작.
  계산용 복사본에는 sync(data) 로 현재 전체 qpos(베이스 자세 포함)를 넣은 뒤 팔 관절만 바꾼다.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float  # m
    ang_err_deg: float

    def ok(self, pos_tol=0.003, ang_tol_deg=3.0) -> bool:
        return self.pos_err <= pos_tol and self.ang_err_deg <= ang_tol_deg


ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]


class PressIK:
    def __init__(self, model: mujoco.MjModel, site: str = "press_tip", body: str = "link6", lam: float = 0.05):
        self.m = model
        self.d = mujoco.MjData(model)  # 계산 전용 복사본
        self.site = model.site(site).id
        self.body = model.body(body).id
        jid = [model.joint(j).id for j in ARM_JOINTS]
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid])
        self.lo = model.jnt_range[jid, 0].copy()
        self.hi = model.jnt_range[jid, 1].copy()
        self.lam = lam
        self.qpos_ref = model.qpos0.copy()  # 팔 이외 관절(베이스·버튼·그리퍼) 값
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def sync(self, data: mujoco.MjData):
        """실제 시뮬레이션의 현재 qpos(베이스 자세 포함)를 기준으로 쓴다."""
        self.qpos_ref = data.qpos.copy()

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """q(6) -> (press_tip 위치, link6 z축)."""
        self.d.qpos[:] = self.qpos_ref
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.xmat[self.body].reshape(3, 3)[:, 2].copy()

    def solve_from(self, target, zdir, q0, iters=400) -> IKResult:
        m, d = self.m, self.d
        d.qpos[:] = self.qpos_ref
        d.qpos[self.qadr] = q0
        I6 = np.eye(6) * self.lam ** 2
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.body].reshape(3, 3)
            p = d.site_xpos[self.site]
            mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
            e = np.r_[target - p, 0.5 * np.cross(R[:, 2], zdir)]
            if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-4:
                break
            J = np.r_[self._jp[:, self.dadr], 0.5 * self._jr[:, self.dadr]]
            dq = J.T @ np.linalg.solve(J @ J.T + I6, e)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)
        mujoco.mj_kinematics(m, d)
        R = d.xmat[self.body].reshape(3, 3)
        pos_err = float(np.linalg.norm(target - d.site_xpos[self.site]))
        ang = float(np.degrees(np.arccos(np.clip(R[:, 2] @ zdir, -1, 1))))
        return IKResult(d.qpos[self.qadr].copy(), pos_err, ang)

    def solve(self, target, zdir, q_seed=None, rng=None, n_starts=30, iters=400) -> IKResult:
        """다중 초기값 IK. q_seed 가 있으면 그것부터 시도하고, 관절 이동량이 작은 해를 약하게 선호."""
        rng = rng if rng is not None else np.random.default_rng(0)
        starts = [] if q_seed is None else [np.asarray(q_seed, float)]
        starts += [rng.uniform(self.lo, self.hi) for _ in range(n_starts)]
        best, best_cost = None, np.inf
        for q0 in starts:
            r = self.solve_from(target, zdir, q0, iters)
            cost = r.pos_err + r.ang_err_deg / 1000
            if q_seed is not None and r.ok():
                cost += 1e-4 * np.linalg.norm(r.q - q_seed)  # 같은 정확도면 가까운 해
            if cost < best_cost:
                best, best_cost = r, cost
        return best
