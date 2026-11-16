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
    return opt


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

    def capture(self):
        s = self.sim
        self.big.update_scene(s.data, camera=self.cam, scene_option=self.opt)
        obs = self.big.render()
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
        for i in (1, 2, 3):
            cur = self.sim.button_disp(i) * 1000
            pk = self.sim.peak[i] * 1000
            dr.rectangle([x0, y + 82, x0 + 105, y + 104], outline=(120, 120, 120))
            dr.rectangle([x0, y + 82, x0 + int(105 * min(max(cur, 0.0), 15) / 15), y + 104], fill=BTN_COLORS[i])
            dr.line([x0 + 70, y + 80, x0 + 70, y + 106], fill=(255, 255, 255))
            f.draw(dr, (x0 + 111, y + 83), f"{i}층 {cur:4.1f} / max {pk:4.1f}mm", (230, 230, 230))
            x0 += 316
        if self.result is not None:
            ok = self.result.get("ok")
            label = self.result.get("label", "SUCCESS" if ok else "FAIL")
            color = (60, 220, 90) if ok else (240, 70, 70)
            dr.rectangle([8, 36, 360, 84], fill=(0, 0, 0))
            self.font_big.draw(dr, (16, 40), label, color)

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
