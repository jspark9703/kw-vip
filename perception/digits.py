"""버튼 숫자 인식 (scene_v3): wrist_rgb RGB-D -> {층: 버튼 앞면 중심, 누르는 방향, 점수}.

버튼 색은 모두 같으므로 '숫자 글씨'로만 층을 구분한다.
1) 관찰 자세: 손목 카메라가 패널 정면 ~24cm 앞에서 패널을 보도록 IK (head 카메라 패널 추정 사용)
2) 손목 depth 로 패널 평면을 다시 맞춘다 (가까워서 head 추정보다 정확)
3) 정면화: 패널 평면 위 격자(0.5mm/px, 오른쪽 = z x n, 위 = 월드 z)를 카메라에 투영해 RGB 를 샘플링
   -> 손목 회전(IK 가 구속하지 않는 축)과 비스듬한 시선이 모두 제거된다
4) 밝은 원(버튼) 검출 -> 중심 = 정면화 좌표 -> 3D
5) 라벨 영역을 이진화해 숫자 템플릿(같은 폰트로 렌더)과 정규화 상호상관(NCC), 작은 이동·크기 변화 탐색
6) 버튼-숫자 1:1 배정(헝가리안). 점수 < SCORE_MIN 이면 '판독 불가'
반환 형식: site_gt.locate_button 과 같은 {"face", "into", "source"} + score
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from sim.recorder import hidden_site_option

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"
WH, WW = 480, 640
MM_PX = 2.0                       # 정면화 해상도 (px/mm)
RECT_W, RECT_H = 0.30, 0.24       # 정면화 범위 (m)
LABEL_OFF = 0.0457                # 패널 앞면 -> 버튼 라벨 면 (m)
FACE_OFF = 0.045                  # 패널 앞면 -> 버튼 앞면 (m)
BTN_R = 0.022
LABEL_HALF = 0.0135
SCORE_MIN = 0.45
OBS_DIST = 0.24                   # 관찰 거리 (라벨 면 -> 카메라)
CAM_AHEAD = 0.056                 # wrist 카메라가 press_tip 보다 접근축 방향으로 앞선 거리


def _templates(size_px: int) -> dict[int, np.ndarray]:
    """라벨과 같은 구성(128px 에 104pt 숫자 가운데)으로 렌더 후 size_px 로 축소, 0/1 (글씨=1)."""
    font = ImageFont.truetype(str(FONT), 104)
    out = {}
    for n in range(1, 10):
        im = Image.new("L", (128, 128), 255)
        dr = ImageDraw.Draw(im)
        l, t, r, b = dr.textbbox((0, 0), str(n), font=font)
        dr.text(((128 - (r - l)) / 2 - l, (128 - (b - t)) / 2 - t), str(n), fill=0, font=font)
        a = np.asarray(im.resize((size_px, size_px), Image.BILINEAR), float) / 255.0
        out[n] = (a < 0.5).astype(float)
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
        """월드 점 -> (u, v) 픽셀 좌표 (실수), 카메라 앞인지 여부."""
        R = data.cam_xmat[self.cam].reshape(3, 3)
        pc = (P - data.cam_xpos[self.cam]) @ R
        z = -pc[:, 2]
        u = WW / 2 + self.f * pc[:, 0] / np.maximum(z, 1e-6) - 0.5
        v = WH / 2 - self.f * pc[:, 1] / np.maximum(z, 1e-6) - 0.5
        return u, v, z > 0.01

    def close(self):
        self.rgb_r.close()
        self.dep_r.close()


def refine_plane(cam: WristCamera, data, depth, face, n, rng=None):
    """손목 depth 에서 패널 앞면 평면 재추정 (3점 RANSAC, 허용 2mm). (앞면 위 점, 바깥 법선).
    후보 점 = 추정 앞면에서 ±2.5cm, 옆으로 13cm 안. 버튼 앞면(+4.5cm)은 범위 밖이라 패널 면이 최대 평면."""
    rng = rng if rng is not None else np.random.default_rng(0)
    vs, us = np.mgrid[0:WH:3, 0:WW:3]
    vs, us = vs.ravel(), us.ravel()
    P = cam.backproject(data, vs, us, depth)
    d = (P - face) @ n
    lat = np.linalg.norm((P - face) - np.outer(d, n), axis=1)
    Q = P[(np.abs(d) < 0.025) & (lat < 0.13)]
    if len(Q) < 200:
        return face, n
    best, best_k = None, 0
    for _ in range(150):
        a, b_, c_ = Q[rng.choice(len(Q), 3, replace=False)]
        nn = np.cross(b_ - a, c_ - a)
        if np.linalg.norm(nn) < 1e-6:
            continue
        nn /= np.linalg.norm(nn)
        if abs(nn @ n) < 0.9:          # head 추정 법선에서 25° 이상 벗어난 평면은 패널 아님
            continue
        k = int(np.sum(np.abs((Q - a) @ nn) < 0.002))
        if k > best_k:
            best_k, best = k, (a, nn)
    if best is None:
        return face, n
    a, nn = best
    inl = Q[np.abs((Q - a) @ nn) < 0.002]
    c = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c)
    n2 = vt[-1] * np.sign(vt[-1] @ n)
    # 패널 앞면 위의 점: 기존 중심(가로·세로)을 새 평면에 투영
    return face - ((face - c) @ n2) * n2, n2


def rectify(cam: WristCamera, data, rgb, plane_pt, n, depth=None, plane_tol=0.008):
    """패널 평면(라벨 면) 위 격자 -> 정면 영상. 반환 (영상 HxWx3, 좌표 함수 px->3D, 유효 마스크).
    depth 를 주면 유효 마스크 = '실제 depth 가 그 평면 위 점과 plane_tol 안에서 일치' (평면 위에 있는 픽셀만)."""
    hz = np.cross([0, 0, 1.0], n)
    hz /= np.linalg.norm(hz)
    up = np.cross(n, hz)
    nw, nh = int(RECT_W * 1000 * MM_PX), int(RECT_H * 1000 * MM_PX)
    a = (np.arange(nw) + 0.5) / (1000 * MM_PX) - RECT_W / 2
    b = RECT_H / 2 - (np.arange(nh) + 0.5) / (1000 * MM_PX)
    A, B = np.meshgrid(a, b)
    P = plane_pt + A.reshape(-1, 1) * hz + B.reshape(-1, 1) * up
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
        return plane_pt + ((col + 0.5) / (1000 * MM_PX) - RECT_W / 2) * hz + (RECT_H / 2 - (row + 0.5) / (1000 * MM_PX)) * up
    return img.reshape(nh, nw, 3), to3d, ok.reshape(nh, nw)


def find_buttons(img, on_plane=None):
    """버튼 앞면 = 라벨 면 위에 있는 픽셀(depth 일치) -> 원. depth 가 없으면 밝기로. 반환 [(col, row, 면적)]."""
    if on_plane is not None:
        m = on_plane.copy()      # 라벨 면 위 픽셀 = 버튼 앞면 (패널 면은 4.5cm 뒤). 그림자 진 버튼도 잡힘
    else:
        m = img.astype(float).mean(2) / 255.0 > 0.45
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    area_exp = np.pi * (BTN_R * 1000 * MM_PX) ** 2
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        rr, cc = np.nonzero(lab[sl] == k)
        a = len(rr)
        if not (0.45 * area_exp < a < 1.6 * area_exp):
            continue
        hgt, wid = rr.max() - rr.min() + 1, cc.max() - cc.min() + 1
        if not (0.7 < hgt / wid < 1.4):
            continue
        out.append((cc.mean() + sl[1].start, rr.mean() + sl[0].start, a))
    return out


def classify(img, buttons, tmpl):
    """각 버튼 라벨 -> 숫자별 NCC 점수 행렬 (버튼 x 숫자 1..9)."""
    g = img.astype(float).mean(2) / 255.0
    half = int(round(LABEL_HALF * 1000 * MM_PX))
    S = np.zeros((len(buttons), 9))
    for i, (cx, cy, _) in enumerate(buttons):
        cx, cy = int(round(cx)), int(round(cy))
        crop = g[max(0, cy - half - 4): cy + half + 5, max(0, cx - half - 4): cx + half + 5]
        if crop.size == 0:
            continue
        ink = (crop < 0.35).astype(float)
        for n, t in tmpl.items():
            best = 0.0
            for s in (0.9, 1.0, 1.1):
                ts = ndimage.zoom(t, s, order=1) if s != 1.0 else t
                th, tw = ts.shape
                for dy in range(0, max(1, ink.shape[0] - th + 1), 2):
                    for dx in range(0, max(1, ink.shape[1] - tw + 1), 2):
                        win = ink[dy:dy + th, dx:dx + tw]
                        if win.shape != ts.shape:
                            continue
                        best = max(best, _ncc(win, ts))
            S[i, n - 1] = best
    return S


def read_panel(cam: WristCamera, data, face0, n0) -> dict:
    """정면화 + 버튼 검출 + 숫자 판독.  face0/n0: 패널 앞면 중심·바깥 법선 추정값."""
    rgb, depth = cam.grab(data)
    face, n = refine_plane(cam, data, depth, np.asarray(face0, float), np.asarray(n0, float))
    img, to3d, on_plane = rectify(cam, data, rgb, face + LABEL_OFF * n, n, depth=depth)
    btns = find_buttons(img, on_plane)
    tmpl = _templates(2 * int(round(LABEL_HALF * 1000 * MM_PX)))
    S = classify(img, btns, tmpl) if btns else np.zeros((0, 9))
    floors = {}
    if len(btns):
        rows, cols = linear_sum_assignment(-S)
        for i, j in zip(rows, cols):
            sc = float(S[i, j])
            if sc < SCORE_MIN:
                continue
            cx, cy, _ = btns[i]
            p_label = to3d(cx, cy)
            floors[j + 1] = {"face": p_label - (LABEL_OFF - FACE_OFF) * n, "into": -n, "score": round(sc, 3),
                             "px": (round(cx, 1), round(cy, 1)), "source": "wrist_digits"}
    return {"floors": floors, "n_buttons": len(btns), "scores": S.round(3), "rect": img, "rgb": rgb,
            "plane": (face, n)}


def lamp_on(cam: WristCamera, data, face_btn, n, save=None) -> tuple[bool, float]:
    """VERIFY: 버튼 앞면(라벨 사각형 제외)이 램프 색(주황)인지. (점등 여부, 주황 픽셀 비율).
    판정은 HSV: 색상 0.03~0.13, 채도 > 0.5 (그림자 진 면도 색상·채도는 유지됨)."""
    from matplotlib.colors import rgb_to_hsv
    rgb, depth = cam.grab(data)
    img, _, ok = rectify(cam, data, rgb, np.asarray(face_btn) + 0.0007 * np.asarray(n), np.asarray(n), depth=depth)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    sx, sy = (xx - w / 2) / (1000 * MM_PX), (yy - h / 2) / (1000 * MM_PX)
    region = (np.hypot(sx, sy) < BTN_R * 0.9) & ((np.abs(sx) > LABEL_HALF * 1.1) | (np.abs(sy) > LABEL_HALF * 1.1)) & ok
    if save is not None:
        Image.fromarray(np.hstack([img, (np.stack([region] * 3, 2) * 255).astype(np.uint8)])).save(save)
    if region.sum() < 20:
        return False, 0.0
    hsv = rgb_to_hsv(img[region] / 255.0)
    orange = (hsv[:, 0] > 0.03) & (hsv[:, 0] < 0.13) & (hsv[:, 1] > 0.5)
    frac = float(orange.mean())
    return frac > 0.4, round(frac, 3)
