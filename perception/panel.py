"""도어락 확인 (scene_v4): head_rgb depth(+RGB 기록) -> 현관 도어락 원점·법선 -> 도킹 목표 자세.

색은 조명에 따라 벽·문과 구분이 잘 안 돼서 '기하'로 찾는다.
1) depth 를 3D 로 복원 (높이 0.6~1.8m 점만)
2) RANSAC 으로 가장 큰 '수직 평면' = 벽(문은 벽과 거의 같은 면, 3mm 앞). 법선은 카메라 쪽
3) 벽 앞으로 1.2~4.5cm 튀어나온 점 = 도어락 후보 (하우징 앞면 2.5cm, 키 2.9cm) -> 화면에서 연결 요소
   (문틀은 가늘고 길고, 손잡이·호수판은 작거나 납작해서 크기로 걸러짐)
4) 크기(가로 0.08 x 세로 0.17m 근처)로 고르고, 후보가 여럿이면(이웃집 도어락) 지도 사전정보 위치에 가장 가까운 것
5) 도어락 원점 = 가로·세로 분위수(2~98%) 중간, 깊이 = 벽 + 3mm(문짝 면) -> DOCK_REL 을 거꾸로 적용해 도킹 자세
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from models.make_scene_v4 import DOCK_REL, HOUSING_HALF, LOCK_REL_NOMINAL
from perception.camera import H, W, HeadCamera, _fit_plane

PANEL_W, PANEL_H = 2 * HOUSING_HALF[0], 2 * HOUSING_HALF[2]
DOOR_FRONT = LOCK_REL_NOMINAL[1]      # 벽 면 -> 문짝 앞면 (도어락 원점이 붙는 면)
BAND = (0.012, 0.045)                 # 벽에서 튀어나온 높이 범위 (m)
SIZE_TOL = 0.035
MIN_PX = 20
PRIOR_MAX = 1.2                       # 사전정보 위치에서 이보다 먼 후보는 이웃집 도어락으로 보고 버림 (m)


def dock_from_panel(center, normal) -> np.ndarray:
    """도어락 원점 + 바깥 법선 -> 도킹 자세 (x, y, yaw). 휠체어 진행 방향 u = 법선을 -90° 돌린 것."""
    n = np.asarray(normal, float)[:2]
    n = n / np.linalg.norm(n)
    u = np.array([n[1], -n[0]])
    yaw = float(np.arctan2(u[1], u[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    rel = np.array([c * DOCK_REL[0] - s * DOCK_REL[1], s * DOCK_REL[0] + c * DOCK_REL[1]])
    xy = np.asarray(center, float)[:2] - rel
    return np.array([xy[0], xy[1], yaw])


def _wall_plane(P, rng, iters=200, tol=0.01):
    """수직 평면 RANSAC: 두 점 + z 축으로 평면을 만든다. (점, 법선, inlier 마스크)."""
    best_n, best = 0, None
    for _ in range(iters):
        i, j = rng.integers(0, len(P), 2)
        d = P[j] - P[i]
        n = np.array([-d[1], d[0], 0.0])
        if np.linalg.norm(n) < 0.2:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - P[i]) @ n) < tol
        k = int(inl.sum())
        if k > best_n:
            best_n, best = k, (P[i], n, inl)
    if best is None:
        return None
    c, n, inl = best
    Q = P[inl]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd((Q - c)[:, :2], full_matrices=False)   # 수평면에서 다시 맞춤
    n = np.array([vt[-1][0], vt[-1][1], 0.0])
    return c, n / np.linalg.norm(n), np.abs((P - c) @ n) < tol


def detect_panel(cam: HeadCamera, data, rng=None, prior=None) -> dict:
    rgb, depth = cam.grab(data)
    rng = rng if rng is not None else np.random.default_rng(0)
    vs, us = np.mgrid[0:H:2, 0:W:2]
    vs, us = vs.ravel(), us.ravel()
    ok = depth[vs, us] < 4.0
    vs, us = vs[ok], us[ok]
    P = cam.backproject(data, vs, us, depth)
    sel = (P[:, 2] > 0.6) & (P[:, 2] < 1.8)
    vs, us, P = vs[sel], us[sel], P[sel]
    out = {"panel": None, "cands": [], "rgb": rgb}
    if len(P) < 500:
        return out
    wall = _wall_plane(P, rng)
    if wall is None:
        return out
    c, n, _ = wall
    if n @ (data.cam_xpos[cam.cam] - c) < 0:
        n = -n
    h = (P - c) @ n
    m = np.zeros((H, W), bool)
    pm = (h > BAND[0]) & (h < BAND[1])
    m[vs[pm], us[pm]] = True
    m = ndimage.binary_closing(ndimage.binary_dilation(m, iterations=1), iterations=3)
    lab, k = ndimage.label(m)
    sizes = np.bincount(lab.ravel(), minlength=k + 1)
    hz = np.cross([0, 0, 1.0], n)
    best = None
    for q in np.nonzero(sizes >= MIN_PX)[0]:
        if q == 0:
            continue
        sl = ndimage.find_objects(lab == q)[0]
        v2, u2 = np.nonzero(lab[sl] == q)
        v2, u2 = v2 + sl[0].start, u2 + sl[1].start
        Q = cam.backproject(data, v2, u2, depth)
        hq = (Q - c) @ n
        Q = Q[(hq > BAND[0]) & (hq < BAND[1])]
        if len(Q) < MIN_PX // 2:
            continue
        a, b = Q @ hz, Q[:, 2]
        a0, a1 = np.percentile(a, [2, 98])
        b0, b1 = np.percentile(b, [2, 98])
        w_, h_ = a1 - a0, b1 - b0
        out["cands"].append({"w": round(float(w_), 3), "h": round(float(h_), 3), "n": int(len(Q))})
        if abs(w_ - PANEL_W) > SIZE_TOL or abs(h_ - PANEL_H) > SIZE_TOL:
            continue
        origin = hz * (a0 + a1) / 2 + np.array([0, 0, (b0 + b1) / 2.0]) + n * (c @ n + DOOR_FRONT)  # (h, n, z) 직교 기저
        score = abs(w_ - PANEL_W) + abs(h_ - PANEL_H)
        if prior is not None:
            dp = float(np.linalg.norm(origin[:2] - np.asarray(prior)[:2]))
            out["cands"][-1]["prior_m"] = round(dp, 2)
            if dp > PRIOR_MAX:
                continue
            score += dp           # 이웃집 도어락보다 우리 집(사전정보 위치에 가까운) 것
        if best is None or score < best["score"]:
            best = {"normal": n, "center": origin, "score": score, "size": (float(w_), float(h_)), "n_px": int(len(Q))}
    out["panel"] = best
    return out


class PanelLocator:
    """head 카메라로 도어락을 찾고 도킹 자세를 낸다. sim.model 당 하나. prior = 지도 사전정보(도어락 대략 위치)."""

    def __init__(self, model, prior=None):
        self.cam = HeadCamera(model)
        self.prior = prior
        self.last = None

    def __call__(self, sim) -> dict | None:
        det = detect_panel(self.cam, sim.data, prior=self.prior)
        p = det["panel"]
        if p is None:
            self.last = {"found": False, "cands": det["cands"]}
            return None
        gt = sim.data.xpos[sim.model.body("doorlock").id]   # 평가 기록용 (판단에는 쓰지 않음)
        err = float(np.linalg.norm(p["center"] - gt) * 1000)
        dk = dock_from_panel(p["center"], p["normal"])
        self.last = {"found": True, "center": p["center"].round(4).tolist(), "normal": p["normal"].round(4).tolist(),
                     "size": [round(v, 3) for v in p["size"]], "err_mm": round(err, 1), "dock": dk.round(4).tolist()}
        return {"center": p["center"], "normal": p["normal"], "dock": dk, "err_mm": err}

    def close(self):
        self.cam.close()
