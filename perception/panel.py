"""패널 확인 (scene_v3): head_rgb depth(+RGB 기록) -> 엘리베이터 패널 중심·법선 -> 도킹 목표 자세.

색은 조명에 따라 벽과 구분이 잘 안 돼서(벽도 저채도·중간 밝기) '기하'로 찾는다.
1) depth 를 3D 로 복원 (높이 0.3~2.2m 점만)
2) RANSAC 으로 가장 큰 '수직 평면' = 벽. 법선은 카메라 쪽
3) 벽 앞으로 4~12cm 튀어나온 점 = 패널 후보 (패널 앞면 7cm, 버튼 ~10cm) -> 화면에서 연결 요소
4) 크기(가로·세로 = 0.26 x 0.20m 근처)로 고르고, 벽 법선 기준 가로·세로 분위수(2~98%) 중간 = 앞면 중심
5) 도킹 자세 = 패널 자세에서 DOCK_REL(휠체어 기준 패널 위치)을 거꾸로 적용
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from models.make_scene_v3 import DOCK_REL
from perception.camera import H, W, HeadCamera, _fit_plane

PANEL_W, PANEL_H, PANEL_HALF_T = 0.26, 0.20, 0.035
SIZE_TOL = 0.06
MIN_PX = 60


def dock_from_panel(center, normal) -> np.ndarray:
    """패널 body 중심 + 바깥 법선 -> 도킹 자세 (x, y, yaw). 휠체어 진행 방향 u = 법선을 -90° 돌린 것."""
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


def detect_panel(cam: HeadCamera, data, rng=None) -> dict:
    rgb, depth = cam.grab(data)
    rng = rng if rng is not None else np.random.default_rng(0)
    vs, us = np.mgrid[0:H:2, 0:W:2]
    vs, us = vs.ravel(), us.ravel()
    ok = depth[vs, us] < 4.0
    vs, us = vs[ok], us[ok]
    P = cam.backproject(data, vs, us, depth)
    sel = (P[:, 2] > 0.3) & (P[:, 2] < 2.2)
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
    pm = (h > 0.04) & (h < 0.12)
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
        Q = Q[(hq > 0.04) & (hq < 0.12)]
        if len(Q) < MIN_PX // 2:
            continue
        a, b = Q @ hz, Q[:, 2]
        a0, a1 = np.percentile(a, [2, 98])
        b0, b1 = np.percentile(b, [2, 98])
        w_, h_ = a1 - a0, b1 - b0
        out["cands"].append({"w": round(float(w_), 3), "h": round(float(h_), 3), "n": int(len(Q))})
        if abs(w_ - PANEL_W) > SIZE_TOL or abs(h_ - PANEL_H) > SIZE_TOL:
            continue
        # 앞면 = 벽에서 패널 두께(7cm) 근처 점들
        front = (Q - c) @ n
        dfront = float(np.median(front[(front > 0.05) & (front < 0.085)])) if np.any((front > 0.05) & (front < 0.085)) else 0.07
        face = hz * (a0 + a1) / 2 + np.array([0, 0, (b0 + b1) / 2.0]) + n * (c @ n + dfront)  # (h, n, z) 직교 기저
        score = abs(w_ - PANEL_W) + abs(h_ - PANEL_H)
        if best is None or score < best["score"]:
            best = {"face": face, "normal": n, "center": face - PANEL_HALF_T * n, "score": score,
                    "size": (float(w_), float(h_)), "n_px": int(len(Q))}
    out["panel"] = best
    return out


class PanelLocator:
    """head 카메라로 패널을 찾고 도킹 자세를 낸다. sim.model 당 하나."""

    def __init__(self, model):
        self.cam = HeadCamera(model)
        self.last = None

    def __call__(self, sim) -> dict | None:
        det = detect_panel(self.cam, sim.data)
        p = det["panel"]
        if p is None:
            self.last = {"found": False, "cands": det["cands"]}
            return None
        gt = sim.data.xpos[sim.model.body("elevator_panel").id]
        err = float(np.linalg.norm(p["center"] - gt) * 1000)
        dk = dock_from_panel(p["center"], p["normal"])
        self.last = {"found": True, "center": p["center"].round(4).tolist(), "normal": p["normal"].round(4).tolist(),
                     "size": [round(v, 3) for v in p["size"]], "err_mm": round(err, 1), "dock": dk.round(4).tolist()}
        return {"center": p["center"], "normal": p["normal"], "dock": dk, "err_mm": err}

    def close(self):
        self.cam.close()
