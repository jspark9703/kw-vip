"""경로 추종 + 재계획 (10Hz 제어 주기, 물리는 매 스텝 진행).

Navigator.go_to(goal)      : 라이다로 격자를 갱신하며 A* 경로를 pure pursuit 로 따라간다.
                             - 진행 방향 오차가 크면 제자리 회전(차동구동)
                             - 새 장애물이 경로를 막으면 / 1s 마다 재계획
                             - 몸체 앞 15cm 상자 안에 라이다 점이 있으면 비상 정지 -> (뒤가 비면 15cm 후진) -> 재계획
                               (그 뒤 3s 는 반경 +12cm 로 먼저 계획, 같은 목표로 6번 쌓이면 대안 목표로)
Navigator.turn_to(yaw)     : 제자리 회전
Navigator.approach_line(p, yaw) : 직선 접근(도킹 마지막 구간). 횡오차를 조향으로 보정하며 목표 x 까지 전진.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nav.grid import ROBOT_R, OccupancyGrid
from nav.lidar import front_blocked, scan_points, sector_min
from nav.planner import astar, path_length
from sim.world import wrap

CTRL_DT = 0.1
WIDE_EXTRA = 0.12      # 비상정지 뒤 3s 동안 계획 반경에 더하는 여유 (m)
ESTOP_SWITCH = 6       # 같은 목표로 가다 비상정지가 이만큼 쌓이면 대안 목표로 바꿈


@dataclass
class NavConfig:
    v_max: float = 0.4
    w_max: float = 0.9
    lookahead: float = 0.5
    turn_in_place: float = 0.7      # rad, 이보다 크면 제자리 회전
    goal_tol: float = 0.06          # m
    replan_every: float = 1.0       # s
    stop_dist: float = 0.70         # 라이다 원점 기준 전방 최소 거리 (몸체 앞 끝 ~0.55m)
    timeout: float = 90.0           # s (시뮬레이션 시간)


@dataclass
class NavLog:
    replans: int = 0
    estops: int = 0
    traveled: float = 0.0
    plans: list = field(default_factory=list)   # 계획된 경로 (미니맵·분석용)
    t: float = 0.0


class Navigator:
    def __init__(self, sim, grid: OccupancyGrid | None = None, cfg: NavConfig | None = None, status=None):
        self.sim = sim
        self.grid = grid or OccupancyGrid()
        self.cfg = cfg or NavConfig()
        self.status = status or (lambda **kw: None)
        self.log = NavLog()
        self.wide_until = -1.0
        self.path = None
        self.n_ctrl = max(1, int(round(CTRL_DT / sim.dt)))
        self.scan()

    # ---------- 공통 ----------
    def scan(self):
        pts, _, r = scan_points(self.sim)
        new = self.grid.add_points(pts)
        self.last_r = r
        return new

    def tick(self, v, w):
        p0 = self.sim.base_pose()
        self.sim.set_twist(float(np.clip(v, -self.cfg.v_max, self.cfg.v_max)),
                           float(np.clip(w, -self.cfg.w_max, self.cfg.w_max)))
        self.sim.step(self.n_ctrl)
        self.log.traveled += float(np.linalg.norm(self.sim.base_pose()[:2] - p0[:2]))
        self.log.t += CTRL_DT
        return self.scan()

    def stop(self, settle=0.3):
        n = int(settle / self.sim.dt)
        self.sim.set_twist(0.0, 0.0)
        self.sim.step(n)
        self.log.t += n * self.sim.dt          # 시간 한도가 비상정지 반복에서도 동작하도록

    def plan(self, goal):
        pos = self.sim.base_pose()[:2]
        self.path = None
        if self.log.t < self.wide_until:   # 비상정지 직후: 장애물에서 더 떨어진 경로를 먼저 시도
            self.path = astar(self.grid, pos, goal, ROBOT_R + WIDE_EXTRA)
        if self.path is None:
            self.path = astar(self.grid, pos, goal, ROBOT_R)
        self.log.replans += 1
        if self.path is not None:
            self.log.plans.append([p.tolist() for p in self.path])
        self.status(path=self.path, grid=self.grid)
        return self.path

    def _path_blocked(self) -> bool:
        return self._blocked(self.path)

    def _blocked(self, path) -> bool:
        if path is None:
            return True
        pos = self.sim.base_pose()[:2]
        pts = [pos] + list(path[1:])
        # 현재 위치 바로 옆 구간은 출발 칸 예외처럼 조금 느슨하게
        return any(not self.grid.segment_free(a, b, ROBOT_R * 0.97) for a, b in zip(pts[1:-1], pts[2:])) or \
            (len(pts) > 1 and not self.grid.segment_free(pos + 0.25 * (pts[1] - pos) / max(1e-6, np.linalg.norm(pts[1] - pos)),
                                                       pts[1], ROBOT_R * 0.97))

    def _lookahead(self, pos):
        """경로 위에서 pos 로부터 lookahead 만큼 앞의 점."""
        path = self.path
        if len(path) < 2:
            return path[-1]
        # 가장 가까운 구간
        best, k_best, t_best = np.inf, 0, 0.0
        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            ab = b - a
            t = np.clip((pos - a) @ ab / max(ab @ ab, 1e-9), 0, 1)
            dd = np.linalg.norm(pos - (a + t * ab))
            if dd < best:
                best, k_best, t_best = dd, k, t
        L = self.cfg.lookahead
        p = path[k_best] + t_best * (path[k_best + 1] - path[k_best])
        k = k_best
        while k < len(path) - 1:
            seg = path[k + 1] - p
            n = np.linalg.norm(seg)
            if n >= L:
                return p + seg / n * L
            L -= n
            p = path[k + 1]
            k += 1
        return path[-1]

    # ---------- 동작 ----------
    def go_to(self, goal, tol=None, alts=()) -> dict:
        """goal 로 주행. goal 로 경로가 없어지면 alts(대안 목표) 중 경로가 있는 것으로 바꾼다.
        반환 {"ok", "why", "goal_idx"}: goal_idx 0 = goal, k = alts[k-1]."""
        c = self.cfg
        tol = tol or c.goal_tol
        goals = [np.asarray(g, float)[:2] for g in (goal, *alts)]
        gi = 0
        t0 = self.log.t
        last_plan, t_noplan = -1e9, None
        self.path = None                     # 이전 목표의 경로는 버린다
        estops_here, banned = 0, set()
        self.status(state="NAV", goal=goals[gi])
        while True:
            pose = self.sim.base_pose()
            pos = pose[:2]
            dist = np.linalg.norm(goals[gi] - pos)
            if dist < tol:
                self.stop()
                return {"ok": True, "why": None, "goal_idx": gi}
            if self.log.t - t0 > c.timeout:
                self.stop()
                return {"ok": False, "why": "nav_timeout", "goal_idx": gi}
            if self.path is None or self.log.t - last_plan >= c.replan_every or self._path_blocked():
                last_plan = self.log.t
                old = self.path
                if gi in banned or self.plan(goals[gi]) is None:
                    self.path = None
                    for k in range(len(goals)):          # 대안 목표
                        if k != gi and k not in banned and self.plan(goals[k]) is not None:
                            gi = k
                            self.status(goal=goals[gi])
                            break
                if self.path is None:
                    t_noplan = t_noplan if t_noplan is not None else self.log.t
                    if self.log.t - t_noplan > 5.0:
                        self.stop()
                        return {"ok": False, "why": "nav_blocked", "goal_idx": gi}
                    self.tick(0.0, 0.0)                 # 멈춰서 더 관찰
                    continue
                t_noplan = None
                # 기존 경로가 여전히 유효하고 새 경로가 확실히 짧지 않으면 유지 (경로가 좌우로 바뀌는 진동 방지)
                if old is not None and self.path is not None and old is not self.path and \
                        np.linalg.norm(old[-1] - self.path[-1]) < 1e-6 and not self._blocked(old) and \
                        path_length([pos, *old[1:]]) < 1.15 * path_length(self.path):
                    self.path = old
            tgt = self._lookahead(pos)
            err = wrap(np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]) - pose[2])
            if abs(err) > c.turn_in_place:
                v, w = 0.0, 1.5 * err
            else:
                v = c.v_max * max(0.0, np.cos(err)) * min(1.0, dist / 0.6 + 0.15)
                k = 2 * np.sin(err) / max(np.linalg.norm(tgt - pos), 1e-3)  # pure pursuit 곡률
                w = v * k + 0.8 * err * (v < 0.05)
            if v > 0 and front_blocked(self.last_r):
                self.log.estops += 1
                estops_here += 1
                self.wide_until = self.log.t + 3.0
                if estops_here >= ESTOP_SWITCH and len(goals) > 1 and len(banned) < len(goals) - 1:
                    banned.add(gi)                  # 이 목표 쪽은 계속 막힘 -> 다음 재계획에서 대안 목표로
                    estops_here = 0
                self.stop(0.2)
                self.scan()
                # 앞이 막혀 있으면 뒤가 비어 있을 때 15cm 후진해서 공간을 만든 뒤 재계획 (같은 자리 비상정지 반복 방지)
                if not front_blocked(self.last_r, margin=0.2, sign=-1.0):
                    for _ in range(5):
                        self.tick(-0.3, 0.0)
                    self.stop(0.2)
                self.path = None             # 다음 주기에 재계획
                continue
            self.tick(v, w)

    def turn_to(self, yaw, tol=np.radians(1.0), timeout=15.0) -> dict:
        t0 = self.log.t
        while True:
            e = wrap(yaw - self.sim.base_pose()[2])
            if abs(e) < tol:
                self.stop(0.2)
                return {"ok": True, "why": None}
            if self.log.t - t0 > timeout:
                self.stop()
                return {"ok": False, "why": "turn_timeout"}
            self.tick(0.0, np.clip(2.0 * e, -0.8, 0.8) if abs(e) > np.radians(4) else np.sign(e) * 0.08)

    def approach_line(self, target, yaw, v=0.15, tol=0.004, timeout=30.0) -> dict:
        """target(x,y) 를 지나는 yaw 방향 직선을 따라 전진/후진해 target 에 멈춘다 (도킹 마지막 구간)."""
        t0 = self.log.t
        u = np.array([np.cos(yaw), np.sin(yaw)])
        nrm = np.array([-u[1], u[0]])
        target = np.asarray(target, float)[:2]
        n_slow = max(1, int(round(0.02 / self.sim.dt)))
        while True:
            pose = self.sim.base_pose()
            along = (target - pose[:2]) @ u          # 남은 거리 (+ 앞)
            lat = (pose[:2] - target) @ nrm          # 횡오차
            if abs(along) < tol:
                self.stop(0.3)
                return {"ok": True, "why": None, "lat_err": float(lat)}
            if self.log.t - t0 > timeout:
                self.stop()
                return {"ok": False, "why": "dock_timeout", "lat_err": float(lat)}
            vv = np.clip(1.2 * along, -v, v)
            if abs(vv) < 0.02:
                vv = np.sign(along) * 0.02
            sgn = 1.0 if vv >= 0 else -1.0
            head = wrap(yaw - pose[2])
            w = 2.5 * head - sgn * 8.0 * lat * max(0.3, min(1.0, abs(vv) / 0.1))
            p0 = pose
            self.sim.set_twist(vv, float(np.clip(w, -0.5, 0.5)))
            self.sim.step(n_slow)          # 50Hz: 마지막 몇 mm 를 지나치지 않도록
            self.log.traveled += float(np.linalg.norm(self.sim.base_pose()[:2] - p0[:2]))
            self.log.t += n_slow * self.sim.dt


def summary(nav: Navigator) -> dict:
    return {"replans": nav.log.replans, "estops": nav.log.estops, "traveled_m": round(nav.log.traveled, 2),
            "nav_t_s": round(nav.log.t, 1),
            "planned_len_m": round(path_length([np.array(p) for p in nav.log.plans[0]]), 2) if nav.log.plans else None}
