"""카메라 인식 (Phase 5): head_rgb RGB + depth 로 버튼 앞면 중심과 누르는 방향을 추정.

1) RGB 를 HSV 로 바꿔 버튼 색(1층 빨강, 2층 초록, 3층 파랑)으로 마스크
2) 연결 요소 중 크기가 버튼 정도(픽셀 수 MIN~MAX)인 것만 후보 (팔 링크의 빨강은 면적이 커서 제외)
3) depth + 카메라 내부 파라미터(fovy)로 픽셀을 3D 로 복원
4) 버튼 주변 고리 영역(패널 면)의 3D 점에 평면을 맞춰 법선 = 누르는 방향의 반대
5) 법선 방향 높이가 앞면(상위 75% 분위수 ±2mm)인 점들의 평균 = 앞면 중심
색별 후보가 여럿이면(팔의 빨강, 휠체어 시트의 파랑 등) '버튼은 8cm 간격 1줄' 사전지식에 가장 잘 맞는 조합을 고른다.

반환 형식은 site_gt.locate_button 과 같다 (+ err_mm: 시뮬레이터 정답과의 오차, 평가 기록용).
site(목표 마커)는 렌더링에서 숨긴다.  관찰 전 팔을 시야 밖 자세로 옮기지 않는다(기본 자세에서 버튼이 보임).
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage

from perception.site_gt import locate_button as _gt
from sim.recorder import hidden_site_option

H, W = 480, 640
HUE = {1: [(0.94, 1.0), (0.0, 0.04)], 2: [(0.30, 0.45)], 3: [(0.53, 0.63)]}
S_MIN, V_MIN = 0.25, 0.10
MIN_PX, MAX_PX = 25, 4000
SPACING = 0.08


class HeadCamera:
    def __init__(self, model, camera="head_rgb"):
        self.m = model
        self.cam = model.camera(camera).id
        self.rgb_r = mujoco.Renderer(model, H, W)
        self.dep_r = mujoco.Renderer(model, H, W)
        self.dep_r.enable_depth_rendering()
        self.opt = hidden_site_option()
        fovy = np.radians(model.cam_fovy[self.cam])
        self.f = 0.5 * H / np.tan(fovy / 2)

    def grab(self, data):
        self.rgb_r.update_scene(data, camera=self.cam, scene_option=self.opt)
        self.dep_r.update_scene(data, camera=self.cam, scene_option=self.opt)
        return self.rgb_r.render(), self.dep_r.render()

    def backproject(self, data, vs, us, depth):
        z = depth[vs, us]
        xc = (us + 0.5 - W / 2) / self.f * z
        yc = -(vs + 0.5 - H / 2) / self.f * z
        pc = np.stack([xc, yc, -z], 1)
        R = data.cam_xmat[self.cam].reshape(3, 3)
        return data.cam_xpos[self.cam] + pc @ R.T

    def close(self):
        self.rgb_r.close()
        self.dep_r.close()


def _color_mask(hsv, floor):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = np.zeros(h.shape, bool)
    for lo, hi in HUE[floor]:
        m |= (h >= lo) & (h <= hi)
    return m & (s >= S_MIN) & (v >= V_MIN)


def _blobs(mask):
    lab, n = ndimage.label(mask)
    out = []
    for k in range(1, n + 1):
        vs, us = np.nonzero(lab == k)
        if MIN_PX <= len(vs) <= MAX_PX:
            out.append((vs, us))
    return out


def _fit_plane(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[-1]


def _robust(vs, us, P, depth):
    """버튼 가장자리에서 배경 depth 를 잡은 픽셀 제거 (중앙 depth ±4cm)."""
    z = depth[vs, us]
    keep = np.abs(z - np.median(z)) < 0.04
    return (vs[keep], us[keep]), P[keep]


def detect_buttons(cam: HeadCamera, data) -> dict:
    """색별 후보 blob 중 '1줄·8cm 간격' 배치에 가장 잘 맞는 (1,2,3) 조합을 고른다."""
    rgb, depth = cam.grab(data)
    hsv = rgb_to_hsv(rgb / 255.0)
    cands = {}
    for f in (1, 2, 3):
        cands[f] = []
        for vs, us in _blobs(_color_mask(hsv, f)):
            (vs, us), P = _robust(vs, us, cam.backproject(data, vs, us, depth), depth)
            if len(vs) >= MIN_PX:
                cands[f].append(((vs, us), P, P.mean(0)))
    chosen, best = {}, np.inf
    for c1, c2, c3 in itertools.product(cands[1] or [None], cands[2] or [None], cands[3] or [None]):
        cs = {f: c for f, c in zip((1, 2, 3), (c1, c2, c3)) if c is not None}
        cost = 0.0
        if 1 in cs and 3 in cs:
            cost += abs(np.linalg.norm(cs[1][2] - cs[3][2]) - 2 * SPACING)
        if 1 in cs and 2 in cs:
            cost += abs(np.linalg.norm(cs[1][2] - cs[2][2]) - SPACING)
        if 2 in cs and 3 in cs:
            cost += abs(np.linalg.norm(cs[2][2] - cs[3][2]) - SPACING)
        if 1 in cs and 2 in cs and 3 in cs:
            cost += np.linalg.norm(cs[2][2] - (cs[1][2] + cs[3][2]) / 2)
        cost += 1.0 * (3 - len(cs))  # 빠진 버튼 벌점
        if cost < best:
            best, chosen = cost, {f: (c[0], c[1]) for f, c in cs.items()}
    return {"rgb": rgb, "depth": depth, "chosen": chosen, "layout_cost": float(best),
            "n_cands": {f: len(cands[f]) for f in (1, 2, 3)}}


def estimate_face(cam, data, det, floor):
    (vs, us), P = det["chosen"][floor]
    depth = det["depth"]
    # 버튼 주변 고리(패널 면) — 다른 버튼 픽셀 제외
    blob = np.zeros((H, W), bool)
    blob[vs, us] = True
    pad = max(6, int(0.5 * np.sqrt(len(vs))))
    ring = ndimage.binary_dilation(blob, iterations=3 * pad) & ~ndimage.binary_dilation(blob, iterations=pad)
    for f, ((v2, u2), _) in det["chosen"].items():
        other = np.zeros((H, W), bool)
        other[v2, u2] = True
        ring &= ~ndimage.binary_dilation(other, iterations=pad)
    rv, ru = np.nonzero(ring)
    if len(rv) < 30:
        return None, {"why": "panel_ring_too_small"}
    Q = cam.backproject(data, rv, ru, depth)
    zc = np.median(depth[rv, ru])
    Q = Q[np.abs(depth[rv, ru] - zc) < 0.05]  # 배경·팔 제외
    c, n = _fit_plane(Q)
    for _ in range(2):  # 간단한 이상치 제거 후 재적합
        r = np.abs((Q - c) @ n)
        Q = Q[r < max(0.004, 2.5 * np.median(r))]
        c, n = _fit_plane(Q)
    to_cam = data.cam_xpos[cam.cam] - c
    if n @ to_cam < 0:
        n = -n  # 법선을 카메라 쪽(바깥)으로
    h = (P - c) @ n
    h_ref = np.percentile(h, 75)  # 앞면(cap)이 다수 -> 상위 분위수 근처가 앞면 높이 (가장자리 이상치에 강건)
    front = P[np.abs(h - h_ref) < 0.002]
    face = front.mean(0)
    return {"face": face, "into": -n}, {"n_px": int(len(vs)), "n_ring": int(len(Q))}


class CameraLocator:
    """execute(..., locate=CameraLocator(model)) 로 넘겨 쓰는 인식기."""

    def __init__(self, model):
        self.cam = HeadCamera(model)
        self.last = None

    def __call__(self, sim, floor):
        det = detect_buttons(self.cam, sim.data)
        gt = _gt(sim, floor)
        if floor not in det["chosen"]:
            self.last = {"face": None, "info": {"why": "button_not_found", "n_cands": det["n_cands"]}}
            return self.last
        est, info = estimate_face(self.cam, sim.data, det, floor)
        if est is None:
            self.last = {"face": None, "info": info}
            return self.last
        err = float(np.linalg.norm(est["face"] - gt["face"]) * 1000)
        ang = float(np.degrees(np.arccos(np.clip(est["into"] @ gt["into"], -1, 1))))
        self.last = {**est, "source": "camera", "err_mm": round(err, 2), "normal_err_deg": round(ang, 2), **info}
        return self.last

    def close(self):
        self.cam.close()
