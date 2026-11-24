"""녹화 + 오버레이.  서버에는 화면이 없으므로 항상 MUJOCO_GL=egl 로 렌더링하고 mp4 로 확인한다.

화면 구성 (960x608):
  [ 관찰용 자유 카메라 640x480 | head_rgb 320x240 / wrist_rgb 320x240 ]
  [ 텍스트 띠 128px: 명령 · 해석 결과 · 상태 · 버튼 변위(mm) · 성공/실패 ]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"
# 한글 글리프용 (앞에서부터 존재하는 것 사용). 저장소에 NanumGothic(OFL) 포함 -> 컨테이너·노트북 공통.
KO_FONTS = [os.environ.get("KWVIP_FONT", ""), str(FONT_DIR / "NanumGothic-Regular.ttf"),
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", "C:/Windows/Fonts/malgun.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
LATIN_FONTS = [str(FONT_DIR / "NanumGothic-Regular.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
BTN_COLORS = {1: (242, 89, 89), 2: (89, 217, 102), 3: (74, 171, 250)}


def _first(paths, size):
    for p in paths:
        if p and Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return None


class MixedFont:
    """글자마다 한글 폰트 / 라틴 폰트를 골라 그린다 (한글 폰트에 영문 글리프가 없는 경우 대비)."""

    def __init__(self, size):
        self.ko = _first(KO_FONTS, size)
        self.la = _first(LATIN_FONTS, size) or self.ko or ImageFont.load_default()
        self.ko = self.ko or self.la

    def _pick(self, ch):
        return self.ko if ("\uac00" <= ch <= "\ud7a3" or "\u3131" <= ch <= "\u318e") else self.la

    def draw(self, dr: ImageDraw.ImageDraw, xy, text: str, fill):
        x, y = xy
        run, font = "", None
        for ch in text:
            f = self._pick(ch)
            if f is not font and run:
                dr.text((x, y), run, fill=fill, font=font)
                x += font.getlength(run)
                run = ""
            font = f
            run += ch
        if run:
            dr.text((x, y), run, fill=fill, font=font)


def hidden_site_option() -> mujoco.MjvOption:
    """site(목표 마커·press_tip) 는 실제 물체가 아니므로 카메라 영상에서 숨긴다."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0  # 라이다 광선 표시 끔 (scene_v3)
    return opt


def follow_camera(sim, cam: mujoco.MjvCamera):
    """scene_v3: 휠체어를 비스듬히 위에서 따라가는 시점 (패널 근처에서는 패널과 손끝이 함께 보이게)."""
    x, y, _ = sim.base_pose()
    panel = sim.data.xpos[sim.model.body("elevator_panel").id]
    near = np.hypot(panel[0] - x, panel[1] - y) < 1.6
    if near:
        cam.lookat[:] = [0.5 * (x + panel[0]) + 0.1, 0.5 * (y + panel[1]) + 0.2, 0.9]
        cam.distance, cam.azimuth, cam.elevation = 2.0, -60, -28
    else:
        cam.lookat[:] = [x + 0.3, y - 0.2, 0.6]
        cam.distance, cam.azimuth, cam.elevation = 3.2, -80, -58   # 맞은편 벽(y=3.6) 안쪽에서 내려다봄


def observer_camera() -> mujoco.MjvCamera:
    """패널 앞쪽 비스듬한 위에서 보는 시점: 버튼 3개와 누르는 손끝이 함께 보인다."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.30, -0.70, 1.00]
    cam.distance = 1.3
    cam.azimuth = -105
    cam.elevation = -25
    return cam


class Recorder:
    def __init__(self, sim, path: str | Path, fps: int = 20, header: dict | None = None):
        self.sim = sim
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.every = max(1, int(round(1.0 / (fps * sim.dt))))
        self.big = mujoco.Renderer(sim.model, 480, 640)
        self.small = mujoco.Renderer(sim.model, 240, 320)
        self.opt = hidden_site_option()
        self.cam = observer_camera()
        self.font = MixedFont(18)
        self.font_big = MixedFont(34)
        self.header = header or {}
        self.state = ""
        self.result = None
        self.mobile = hasattr(sim, "base_pose")
        self.nav_path = None
        self.grid = None
        self.goal = None
        self.frames: list[np.ndarray] = []
        self._k = 0

    # sim.step_hooks 에 등록
    def __call__(self, sim):
        self._k += 1
        if self._k % self.every == 0:
            self.capture()

    def set_status(self, **kw):
        if "state" in kw:
            self.state = kw["state"]
        if "result" in kw:
            self.result = kw["result"]
        if "path" in kw:
            self.nav_path = kw["path"]
        if "grid" in kw:
            self.grid = kw["grid"]
        if "goal" in kw:
            self.goal = kw["goal"]

    def capture(self):
        s = self.sim
        if self.mobile:
            follow_camera(s, self.cam)
        self.big.update_scene(s.data, camera=self.cam, scene_option=self.opt)
        obs = self.big.render()
        if self.mobile:
            obs = self._minimap(obs)
        self.small.update_scene(s.data, camera="head_rgb", scene_option=self.opt)
        head = self.small.render()
        self.small.update_scene(s.data, camera="wrist_rgb", scene_option=self.opt)
        wrist = self.small.render()
        top = np.hstack([obs, np.vstack([head, wrist])])
        canvas = Image.new("RGB", (960, 608), (18, 20, 24))
        canvas.paste(Image.fromarray(top), (0, 0))
        self._overlay(ImageDraw.Draw(canvas))
        self.frames.append(np.asarray(canvas))

    def _overlay(self, dr: ImageDraw.ImageDraw):
        h, f = self.header, self.font
        f.draw(dr, (648, 4), "head_rgb", (255, 255, 0))
        f.draw(dr, (648, 244), "wrist_rgb", (255, 255, 0))
        f.draw(dr, (8, 4), f"seed {h.get('seed')}  cond {h.get('condition')}  t={self.sim.time:5.2f}s", (255, 255, 255))
        y = 486
        f.draw(dr, (10, y), f"명령: {h.get('command', '')}", (255, 255, 255))
        parsed = h.get("parsed")
        ptxt = json.dumps(parsed, ensure_ascii=False) if parsed is not None else "-"
        f.draw(dr, (10, y + 26), f"해석[{h.get('parser', '-')}]: {ptxt}"[:90], (180, 220, 255))
        f.draw(dr, (10, y + 52), f"실행부: {h.get('runner', '-')}   상태: {self.state}", (255, 220, 140))
        x0 = 10  # 버튼 변위 막대 (흰 세로선 = 10mm 성공 기준)
        floors = self.sim.floors
        bw, step = (105, 316) if len(floors) <= 3 else (52, 158)
        for i in floors:
            cur = self.sim.button_disp(i) * 1000
            pk = self.sim.peak[i] * 1000
            col = BTN_COLORS.get(i, (230, 160, 50)) if len(floors) <= 3 else (230, 160, 50)
            dr.rectangle([x0, y + 82, x0 + bw, y + 104], outline=(120, 120, 120))
            dr.rectangle([x0, y + 82, x0 + int(bw * min(max(cur, 0.0), 15) / 15), y + 104], fill=col)
            dr.line([x0 + int(bw * 10 / 15), y + 80, x0 + int(bw * 10 / 15), y + 106], fill=(255, 255, 255))
            txt = f"{i}층 {cur:4.1f} / max {pk:4.1f}mm" if len(floors) <= 3 else f"{i}층 {pk:4.1f}"
            f.draw(dr, (x0 + bw + 6, y + 83), txt, (230, 230, 230))
            x0 += step
        if self.mobile:
            j = self.sim.n_collision_steps
            f.draw(dr, (420, y + 52), f"충돌 {'없음' if j == 0 else str(j) + ' steps'}", (255, 120, 120) if j else (160, 230, 160))
        if self.result is not None:
            ok = self.result.get("ok")
            label = self.result.get("label", "SUCCESS" if ok else "FAIL")
            color = (60, 220, 90) if ok else (240, 70, 70)
            dr.rectangle([8, 36, 360, 84], fill=(0, 0, 0))
            self.font_big.draw(dr, (16, 40), label, color)

    def _minimap(self, obs):
        """오른쪽 위 미니맵: 라이다 점유격자(검정) + 계획 경로(파랑) + 휠체어(빨강) + 목표(주황)."""
        sim = self.sim
        x0, x1, y0, y1 = -1.0, 8.0, -0.6, 4.2
        W_, H_ = 216, int(216 * (y1 - y0) / (x1 - x0))
        mm = Image.new("RGB", (W_, H_), (245, 245, 245))
        dr = ImageDraw.Draw(mm)

        def px(p):
            return ((p[0] - x0) / (x1 - x0) * W_, (y1 - p[1]) / (y1 - y0) * H_)
        if self.grid is not None:
            occ = self.grid.occ.T[::-1]
            im = Image.fromarray(np.where(occ, 40, 245).astype(np.uint8)).resize((W_, H_), Image.NEAREST)
            mm = Image.merge("RGB", [im] * 3)
            dr = ImageDraw.Draw(mm)
        if self.nav_path is not None and len(self.nav_path) > 1:
            dr.line([px(p) for p in self.nav_path], fill=(0, 90, 255), width=2)
        if self.goal is not None:
            gx, gy = px(self.goal)
            dr.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], outline=(255, 140, 0), width=2)
        x, y, th = sim.base_pose()
        cx, cy = px((x, y))
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(220, 40, 40))
        hx, hy = px((x + 0.4 * np.cos(th), y + 0.4 * np.sin(th)))
        dr.line([cx, cy, hx, hy], fill=(220, 40, 40), width=2)
        out = obs.copy()
        out[4:4 + H_, 640 - W_ - 4:640 - 4] = np.asarray(mm)
        return out

    def hold(self, seconds: float = 1.5):
        """마지막 화면(결과 표시)을 잠시 유지."""
        self.capture()
        self.frames += [self.frames[-1]] * int(seconds * self.fps)

    def close(self) -> str | None:
        try:
            if self.frames:
                import imageio.v2 as imageio
                imageio.mimwrite(self.path, self.frames, fps=self.fps, codec="libx264",
                                 quality=7, macro_block_size=16)
        finally:
            self.big.close()
            self.small.close()
        return str(self.path) if self.frames else None
