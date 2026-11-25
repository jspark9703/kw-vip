"""도어락 키 글자 인식 (scene_v4): wrist_rgb RGB-D -> {키: 키 앞면 중심, 누르는 방향, 점수}.

키 배열은 실제 도어락처럼 고정이지만, 위치를 외워 두지 않고 매번 '글자'로 키를 찾는다.
1) 관찰 자세: 손목 카메라가 키패드 정면 ~12cm 앞에서 보도록 IK (head 카메라의 도어락 추정 사용)
2) 손목 depth 로 하우징 앞면 평면을 다시 맞춘다 (3점 RANSAC, 추정 앞면 ±1cm 만: 2.2cm 뒤의 문짝 면을 피함)
3) 정면화: 키 라벨 면 위 격자(0.25mm/px, 오른쪽 = z x n, 위 = 월드 z)를 카메라에 투영해 RGB 를 샘플링
4) depth 가 라벨 면과 일치하는 픽셀(하우징 면은 4mm 뒤) = 키 앞면 -> 15mm 정사각 덩어리
5) 라벨(흐린 회색조)을 글자 템플릿(같은 폰트·굵기, 0~9 * #)과 정규화 상호상관(NCC), 작은 이동·크기 변화 탐색
6) 키-글자 1:1 배정(헝가리안). 점수 < SCORE_MIN 이면 '판독 불가'
반환 형식: site_gt.locate_key 와 같은 {"face", "into", "source"} + score
VERIFY: 상태 LED 색(초록 = 열림, 빨강 = 틀림)을 같은 정면화로 읽는다.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from models.make_scene_v4 import HOUSING_FRONT, KEY_FACE, KEY_FONT_PX, KEY_HALF, KEY_STROKE, KEY_ZS, KEYS, LED_Z
from sim.recorder import hidden_site_option

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"
WH, WW = 480, 640
MM_PX = 4.0                       # 정면화 해상도 (px/mm)
RECT_W, RECT_H = 0.10, 0.16       # 정면화 범위 (m), 키패드 중심 기준
PAD_UP = 0.010                    # 도어락 원점 -> 키패드 중심 (위로)
LABEL_EPS = 0.0004                # 키 앞면 -> 라벨 면
SCORE_MIN = 0.45
OBS_DIST = 0.12                   # 관찰 거리 (라벨 면 -> 카메라)
CAM_AHEAD = 0.041                 # wrist 카메라가 press_tip(스타일러스 끝)보다 접근축 방향으로 앞선 거리
PLANE_TOL = 0.0018                # 라벨 면 일치 허용 (m): 하우징 면(4mm 뒤)과 구분


def _templates(size_px: int) -> dict[str, np.ndarray]:
    """라벨과 같은 구성(128px, 같은 글자 크기·외곽선)으로 렌더 후 size_px 로 축소한 회색조 (글자 = 밝음)."""
    font = ImageFont.truetype(str(FONT), KEY_FONT_PX)
    out = {}
    for k in KEYS:
        im = Image.new("L", (128, 128), 0)
        dr = ImageDraw.Draw(im)
        l, t, r, b = dr.textbbox((0, 0), k, font=font, stroke_width=KEY_STROKE)
        dr.text(((128 - (r - l)) / 2 - l, (128 - (b - t)) / 2 - t), k, fill=255, font=font,
                stroke_width=KEY_STROKE, stroke_fill=255)
        a = np.asarray(im.resize((size_px, size_px), Image.LANCZOS), float) / 255.0
        out[k] = ndimage.gaussian_filter(a, 1.0)
    return out


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


class WristCamera:
    def __init__(self, model, camera="wrist_rgb"):
        self.m = model
        self.cam = model.camera(camera).id
        self.rgb_r = mujoco.Renderer(model, WH, WW)
        self.dep_r = mujoco.Renderer(model, WH, WW)
        self.dep_r.enable_depth_rendering()
        self.opt = hidden_site_option()
        self.f = 0.5 * WH / np.tan(np.radians(model.cam_fovy[self.cam]) / 2)

    def grab(self, data):
        self.rgb_r.update_scene(data, camera=self.cam, scene_option=self.opt)
        self.dep_r.update_scene(data, camera=self.cam, scene_option=self.opt)
        return self.rgb_r.render(), self.dep_r.render()

    def backproject(self, data, vs, us, depth):
        z = depth[vs, us]
        pc = np.stack([(us + 0.5 - WW / 2) / self.f * z, -(vs + 0.5 - WH / 2) / self.f * z, -z], 1)
        R = data.cam_xmat[self.cam].reshape(3, 3)
        return data.cam_xpos[self.cam] + pc @ R.T

    def project(self, data, P):
        R = data.cam_xmat[self.cam].reshape(3, 3)
        pc = (P - data.cam_xpos[self.cam]) @ R
        z = -pc[:, 2]
        u = WW / 2 + self.f * pc[:, 0] / np.maximum(z, 1e-6) - 0.5
        v = WH / 2 - self.f * pc[:, 1] / np.maximum(z, 1e-6) - 0.5
        return u, v, z > 0.01

    def close(self):
        self.rgb_r.close()
        self.dep_r.close()


def _basis(n):
    hz = np.cross([0, 0, 1.0], n)
    hz /= np.linalg.norm(hz)
    return hz, np.cross(n, hz)


def refine_plane(cam: WristCamera, data, depth, face, n, rng=None):
    """하우징 앞면 평면 재추정 (3점 RANSAC, 허용 1.5mm). 후보 = 추정 앞면 ±1cm, 옆으로 9cm 안. (앞면 위 점, 법선)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    vs, us = np.mgrid[0:WH:2, 0:WW:2]
    vs, us = vs.ravel(), us.ravel()
    P = cam.backproject(data, vs, us, depth)
    d = (P - face) @ n
    lat = np.linalg.norm((P - face) - np.outer(d, n), axis=1)
    Q = P[(np.abs(d) < 0.010) & (lat < 0.09)]
    if len(Q) < 200:
        return face, n
    best, best_k = None, 0
    for _ in range(150):
        a, b_, c_ = Q[rng.choice(len(Q), 3, replace=False)]
        nn = np.cross(b_ - a, c_ - a)
        if np.linalg.norm(nn) < 1e-6:
            continue
        nn /= np.linalg.norm(nn)
        if abs(nn @ n) < 0.9:
            continue
        k = int(np.sum(np.abs((Q - a) @ nn) < 0.0015))
        if k > best_k:
            best_k, best = k, (a, nn)
    if best is None:
        return face, n
    a, nn = best
    inl = Q[np.abs((Q - a) @ nn) < 0.0015]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n2 = vt[-1] * np.sign(vt[-1] @ n)
    return face - ((face - c) @ n2) * n2, n2


def rectify(cam: WristCamera, data, rgb, center, n, w=RECT_W, h=RECT_H, depth=None, plane_tol=PLANE_TOL):
    """평면(center, n) 위 격자 -> 정면 영상. 반환 (영상, 좌표 함수 px->3D, 유효 마스크).
    depth 를 주면 유효 마스크 = 실제 depth 가 그 평면 위 점과 plane_tol 안에서 일치하는 픽셀."""
    hz, up = _basis(n)
    nw, nh = int(w * 1000 * MM_PX), int(h * 1000 * MM_PX)
    a = (np.arange(nw) + 0.5) / (1000 * MM_PX) - w / 2
    b = h / 2 - (np.arange(nh) + 0.5) / (1000 * MM_PX)
    A, B = np.meshgrid(a, b)
    P = center + A.reshape(-1, 1) * hz + B.reshape(-1, 1) * up
    u, v, ok = cam.project(data, P)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok &= (ui >= 0) & (ui < WW) & (vi >= 0) & (vi < WH)
    img = np.zeros((nh * nw, 3), np.uint8)
    img[ok] = rgb[vi[ok], ui[ok]]
    if depth is not None:
        R = data.cam_xmat[cam.cam].reshape(3, 3)
        zexp = -((P - data.cam_xpos[cam.cam]) @ R)[:, 2]
        on = np.zeros(len(P), bool)
        on[ok] = np.abs(depth[vi[ok], ui[ok]] - zexp[ok]) < plane_tol
        ok = on

    def to3d(col, row):
        return center + ((col + 0.5) / (1000 * MM_PX) - w / 2) * hz + (h / 2 - (row + 0.5) / (1000 * MM_PX)) * up
    return img.reshape(nh, nw, 3), to3d, ok.reshape(nh, nw)


def find_keys(on_plane):
    """라벨 면 위 픽셀 덩어리 중 15mm 정사각 크기 = 키. 반환 [(col, row, 면적)]."""
    m = ndimage.binary_opening(ndimage.binary_fill_holes(ndimage.binary_closing(on_plane, iterations=2)), iterations=2)
    lab, _ = ndimage.label(m)
    area_exp = (2 * KEY_HALF * 1000 * MM_PX) ** 2
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == k)
        a = len(rr)
        if not (0.45 * area_exp < a < 1.5 * area_exp):
            continue
        hgt, wid = rr.max() - rr.min() + 1, cc.max() - cc.min() + 1
        if not (0.7 < hgt / wid < 1.4):
            continue
        out.append((cc.mean() + sl[1].start, rr.mean() + sl[0].start, a))
    return out


def classify(img, keys, tmpl):
    """각 키 라벨 -> 글자별 NCC 점수 행렬 (키 x 글자 KEYS 순서). 회색조 NCC (흰 글자가 밝은 쪽)."""
    g = img.astype(float).mean(2) / 255.0
    half = int(round((KEY_HALF - 0.0006) * 1000 * MM_PX))
    S = np.zeros((len(keys), len(KEYS)))
    for i, (cx, cy, _) in enumerate(keys):
        cx, cy = int(round(cx)), int(round(cy))
        crop = g[max(0, cy - half - 6): cy + half + 7, max(0, cx - half - 6): cx + half + 7]
        if crop.size == 0:
            continue
        ink = ndimage.gaussian_filter(crop, 1.0)      # 회색조 그대로 (글자 = 밝음), 템플릿과 같은 흐림
        for j, k in enumerate(KEYS):
            t = tmpl[k]
            best = 0.0
            for s in (0.9, 1.0, 1.1):
                ts = ndimage.zoom(t, s, order=1) if s != 1.0 else t
                th, tw = ts.shape
                for dy in range(0, max(1, ink.shape[0] - th + 1), 2):
                    for dx in range(0, max(1, ink.shape[1] - tw + 1), 2):
                        win = ink[dy:dy + th, dx:dx + tw]
                        if win.shape == ts.shape:
                            best = max(best, _ncc(win, ts))
            S[i, j] = best
    return S


def read_keypad(cam: WristCamera, data, origin, n0) -> dict:
    """정면화 + 키 검출 + 글자 판독.  origin/n0: 도어락 원점(문짝 면 위)·바깥 법선 추정값 (head 카메라)."""
    rgb, depth = cam.grab(data)
    n0 = np.asarray(n0, float)
    _, up0 = _basis(n0)
    front0 = np.asarray(origin, float) + HOUSING_FRONT * n0 + PAD_UP * up0
    front, n = refine_plane(cam, data, depth, front0, n0)
    _, up = _basis(n)
    label_c = front + (KEY_FACE - HOUSING_FRONT + LABEL_EPS) * n
    img, to3d, on_plane = rectify(cam, data, rgb, label_c, n, depth=depth)
    keys = find_keys(on_plane)
    tmpl = _templates(2 * int(round((KEY_HALF - 0.0006) * 1000 * MM_PX)))
    S = classify(img, keys, tmpl) if keys else np.zeros((0, len(KEYS)))
    found = {}
    if len(keys):
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            sc = float(S[i, j])
            if sc < SCORE_MIN:
                continue
            cx, cy, _ = keys[i]
            found[KEYS[j]] = {"face": to3d(cx, cy) - LABEL_EPS * n, "into": -n, "score": round(sc, 3),
                              "px": (round(cx, 1), round(cy, 1)), "source": "wrist_keys"}
    origin_est = front - HOUSING_FRONT * n - PAD_UP * up
    return {"keys": found, "n_keys": len(keys), "scores": S.round(3), "rect": img, "rgb": rgb,
            "plane": (front, n), "origin": origin_est, "normal": n}


def led_state(cam: WristCamera, data, origin, n, save=None, keys=None) -> tuple[str, float]:
    """VERIFY: 상태 LED 색. ('open' 초록 | 'wrong' 빨강 | 'idle' | 'unseen', LED 면적 대비 색 픽셀 비율).
    LED 위치: 판독한 '0' 키가 있으면 그 키 기준(키 위치 오차 <1mm), 없으면 도어락 원점 추정 기준.
    판정: 30x20mm 패치에서 초록(또는 빨강) 픽셀 수가 LED 면적(24x7mm)의 40% 이상."""
    n = np.asarray(n, float)
    _, up = _basis(n)
    if keys and "0" in keys:
        c = np.asarray(keys["0"]["face"]) - (KEY_FACE - HOUSING_FRONT - 0.0006) * n + (LED_Z - KEY_ZS[3]) * up
    else:
        c = np.asarray(origin, float) + (HOUSING_FRONT + 0.0006) * n + LED_Z * up
    rgb, depth = cam.grab(data)
    img, _, ok = rectify(cam, data, rgb, c, n, w=0.030, h=0.020, depth=depth, plane_tol=0.004)
    if save is not None:
        Image.fromarray(img).save(save)
    if ok.sum() < 20:
        return "unseen", 0.0
    hsv = rgb_to_hsv(img / 255.0)
    sat = ok & (hsv[..., 1] > 0.5) & (hsv[..., 2] > 0.2)
    green = sat & (hsv[..., 0] > 0.22) & (hsv[..., 0] < 0.45)
    red = sat & ((hsv[..., 0] < 0.05) | (hsv[..., 0] > 0.95))
    area = (0.024 * 1000 * MM_PX) * (0.007 * 1000 * MM_PX)
    fg, fr = float(green.sum() / area), float(red.sum() / area)
    if fg > 0.4:
        return "open", round(fg, 3)
    if fr > 0.4:
        return "wrong", round(fr, 3)
    return "idle", round(max(fg, fr), 3)
