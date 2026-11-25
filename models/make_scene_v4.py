# 사용법: python models/make_scene_v4.py [models/orig/wheelchair_piper_lite.xml] [models/scene_v4.xml]
"""원본 XML -> scene_v4.xml : 주행 휠체어 + 아파트 복도 + 현관문 도어락(3x4 키패드) + 고정 장애물 + 라이다.

변경점 (발표에서 설명할 것)
 1. 주행: wheelchair_base 에 base_x / base_y (slide, 월드 축) + base_yaw (hinge z) 관절, 속도 액추에이터 3개.
    제어기가 (v, ω) -> (v cosθ, v sinθ, ω) 로 넣어 차동구동(옆 미끄러짐 없음) 제약을 건다. z 는 고정(바닥 접촉 없음).
    질량 120kg 근사 관성. 액추에이터 순서는 팔 7개 다음 -> ctrl[0:7] 의미는 v2 와 같다.
 2. 누름 스타일러스: link6 끝(손가락 패드 앞)에 캡슐 스타일러스(r 4mm, 패드보다 15mm 앞)를 달고 press_tip 을 그 끝으로 옮김.
    손가락 패드 면적(24x18mm)이 도어락 키 간격(20mm)보다 커서 옆 키를 함께 누르기 때문. 접촉력 touch 센서도 스타일러스 끝.
 3. 충돌 비트: 바닥 1 / 휠체어 2 / 벽·장애물 4.  휠체어는 바닥과 충돌하지 않고 벽·장애물과 충돌(충돌 수 집계).
 4. 아파트 복도: 우리 집 현관문(home_door: 문짝·문틀·손잡이·호수판·도어락이 한 몸) + 이웃집 문 3개(가짜 도어락),
    타일 줄눈·걸레받이·맞은편 벽 징두리·천장·천장등·복도 끝 창·소화전함 (모두 시각용, 충돌 없음).
    (문이 있는 벽의 징두리·몰딩은 문 위치가 시드마다 바뀌어 문 앞을 가리므로 넣지 않음)
 5. 도어락: 하우징 80x170mm, 키 12개(1~9 / * 0 #, 실제 배열 고정) 15mm 정사각, 간격 20mm, 누름 깊이 4mm.
    키 앞면 라벨 텍스처(검은 키 + 흰 글씨)로만 구분. 상태 LED(sim 이 대기 회색 / 열림 초록 / 틀림 빨강으로 바꿈).
 6. 센서: 라이다(rangefinder 72빔, 5° 간격, 높이 0.25m: 택배 상자 0.36m 도 보이게), 키 jointpos 12개, 스타일러스 touch 1개.
 7. 장애물 풀 obs_0..5: 택배 상자(정육면체 36~42cm, 테이프 띠) 3개, 사람(원기둥 r 0.20~0.24m, 키 ~1.7m, 머리) 3개.
    쓰지 않는 것은 장면 밖(x=20)에 둔다.
 8. 카메라 near plane: <statistic extent=5>, znear 0.002 (=1cm) — 장면 밖 장애물 때문에 손목 카메라가 가까운 도어락을 잘라내던 문제.
 9. 원본의 테이블·큐브·엘리베이터 버튼은 제거.
"""
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "orig" / "wheelchair_piper_lite.xml"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "scene_v4.xml"
TEX_DIR = ROOT / "assets" / "textures"
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"

# ---- 배치 상수 (sim/world.py · perception 과 공유) ----
DOOR_NOMINAL_X = 5.0                    # 우리 집 현관문 중심 x (벽 앞면 y=0)
LOCK_REL_NOMINAL = (-0.33, 0.003, 1.14)  # 문 중심 기준 도어락 하우징 원점 (손잡이 쪽 가장자리, 높이)
HOUSING_HALF = (0.040, 0.011, 0.085)    # 하우징 반치수 (x, 두께, z)
HOUSING_FRONT = 2 * HOUSING_HALF[1]     # 도어락 원점 -> 하우징 앞면 (m)
KEY_HALF = 0.0075                       # 키 앞면 반폭 (15mm 정사각)
KEY_TRAVEL = 0.004                      # 누름 깊이 한도 (m)
KEY_PROTRUDE = 0.004                    # 하우징 앞면에서 키 앞면까지
KEY_FACE = HOUSING_FRONT + KEY_PROTRUDE  # 도어락 원점 -> 키 앞면 (y)
KEY_PITCH = 0.020
KEY_ROWS = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["*", "0", "#"]]
KEY_ZS = [0.040, 0.020, 0.000, -0.020]  # 행별 키 중심 z (도어락 원점 기준)
KEY_XS = [KEY_PITCH, 0.0, -KEY_PITCH]   # 열별 키 중심 x: 문 앞에서(−y 를 보며) 왼쪽 -> 오른쪽 = +x -> −x
KEYS = [k for row in KEY_ROWS for k in row]
KEY_NAME = {k: {"*": "star", "#": "hash"}.get(k, k) for k in KEYS}   # 관절·몸체 이름용
LED_Z = -0.052
# 휠체어 기준 도어락 원점 위치: 키 앞면이 v2/v3 에서 검증된 버튼 앞면과 같은 '팔 기준' 자리에 오도록
#   (v3 버튼 앞면 = 휠체어 기준 (0.37, -0.79, 1.09)) -> y = -0.79 - KEY_FACE
DOCK_REL = (0.37, -0.79 - KEY_FACE, 1.08)
START_NOMINAL = (0.8, 1.8, 0.0)          # 출발 (x, y, yaw)
KEY_RGBA = ".10 .11 .12 1"
KEY_LIT_RGBA = (0.55, 0.75, 1.0, 1.0)    # 누른 키 백라이트(파랑)
LED_IDLE = (0.25, 0.26, 0.27, 1.0)
LED_OPEN = (0.10, 0.85, 0.30, 1.0)
LED_WRONG = (0.95, 0.15, 0.10, 1.0)
N_LIDAR = 72
LIDAR_POS = (0.06, -0.03, 0.25)
OBSTACLES = [  # (kind, size, rgba): 택배 상자(정육면체) / 사람(원기둥)
    ("parcel", ".20 .20 .20", ".62 .43 .24 1"),
    ("person", ".22 .80", ".24 .34 .50 1"),
    ("parcel", ".18 .18 .18", ".55 .38 .20 1"),
    ("person", ".20 .84", ".36 .30 .28 1"),
    ("parcel", ".21 .21 .21", ".70 .52 .30 1"),
    ("person", ".24 .78", ".20 .40 .32 1"),
]
PARK = (20.0, 20.0)
UNIT = "1203"
NEIGHBORS = [  # (호수, 벽 'near'(y=0) | 'far'(y=3.6), 문 중심 x)
    ("1202", "near", 2.0), ("1205", "far", 2.9), ("1204", "far", 5.9),
]


KEY_FONT_PX, KEY_STROKE = 92, 5        # 키 글자 크기·외곽선 두께 (128px 라벨 기준). perception/digits.py 템플릿도 같은 값


def _text_png(path, text, size, bg, fg, font_px, mirror=True, stroke=0):
    font = ImageFont.truetype(str(FONT), font_px)
    im = Image.new("RGB", size, bg)
    dr = ImageDraw.Draw(im)
    l, t, r, b = dr.textbbox((0, 0), text, font=font, stroke_width=stroke)
    dr.text(((size[0] - (r - l)) / 2 - l, (size[1] - (b - t)) / 2 - t), text, fill=fg, font=font,
            stroke_width=stroke, stroke_fill=fg)
    # MuJoCo 박스 +y 면에 2d 텍스처가 좌우 반전되어 입혀지므로 미리 뒤집어 저장 (렌더링에서 정상으로 보임)
    (im.transpose(Image.FLIP_LEFT_RIGHT) if mirror else im).save(path)


def make_textures():
    """키 라벨(검은 바탕 + 흰 글씨, 128px) · 호수판. 같은 폰트로 인식 템플릿도 만든다(perception/digits.py)."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    bg = tuple(int(float(v) * 255) for v in KEY_RGBA.split()[:3])
    for k in KEYS:
        _text_png(TEX_DIR / f"key_{KEY_NAME[k]}.png", k, (128, 128), bg, (240, 240, 240), KEY_FONT_PX, stroke=KEY_STROKE)
    for unit in [UNIT] + [n[0] for n in NEIGHBORS]:
        _text_png(TEX_DIR / f"unit_{unit}.png", unit, (256, 112), (200, 190, 160), (40, 35, 30), 84)
    _text_png(TEX_DIR / "fire.png", "소화전", (256, 112), (200, 30, 30), (250, 250, 250), 70)


def door_geoms(prefix, unit, facing=1):
    """현관문 한 세트(시각용). facing=+1: 앞면이 +y (near 벽), -1: 앞면이 -y (far 벽). 좌표는 문 중심 기준."""
    s = facing
    return f'''      <geom name="{prefix}_leaf" type="box" pos="0 {s * -.005} 1.05" size=".46 .008 1.05" rgba=".36 .21 .12 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_inset_upper" type="box" pos="0 {s * .004} 1.62" size=".32 .002 .36" rgba=".41 .25 .15 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_inset_lower" type="box" pos="0 {s * .004} .58" size=".32 .002 .42" rgba=".41 .25 .15 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_frame_l" type="box" pos=".50 {s * .0} 1.08" size=".04 .02 1.08" rgba=".80 .77 .70 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_frame_r" type="box" pos="-.50 {s * .0} 1.08" size=".04 .02 1.08" rgba=".80 .77 .70 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_frame_t" type="box" pos="0 {s * .0} 2.14" size=".54 .02 .04" rgba=".80 .77 .70 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_peephole" type="cylinder" pos="0 {s * .006} 1.62" quat=".7071068 .7071068 0 0" size=".02 .004" rgba=".12 .13 .14 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_plate" type="box" pos="0 {s * .023} 2.30" size=".09 .003 .04" material="unit_{unit}" contype="0" conaffinity="0"/>
'''


def neighbor_lock(prefix, facing):
    """이웃집 도어락(가짜: 키 없이 판만). 위치는 문 중심 기준 손잡이 쪽."""
    s = facing
    x = -0.33 * s
    return f'''      <geom name="{prefix}_lock" type="box" pos="{x} {s * .014} 1.15" size=".04 .011 .085" rgba=".14 .15 .16 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_lock_pad" type="box" pos="{x} {s * .026} 1.16" size=".032 .001 .045" rgba=".06 .07 .08 1" contype="0" conaffinity="0"/>
      <geom name="{prefix}_handle" type="box" pos="{x + .03 * s} {s * .045} .99" size=".06 .012 .012" rgba=".72 .70 .66 1" contype="0" conaffinity="0"/>
'''


def build() -> str:
    src = SRC.read_text()
    # 0) 텍스처 경로 · 장면 범위
    src = src.replace('<compiler angle="radian" autolimits="true"/>',
                      '<compiler angle="radian" autolimits="true" texturedir="../assets/textures"/>')
    src = src.replace('<mujoco model="wheelchair_piper_lite">', '<mujoco model="wheelchair_piper_v4">')
    src = src.replace('  <default>', '  <statistic extent="5" center="3.5 1.8 1"/>\n  <visual>\n    <map znear="0.002"/>\n'
                      '    <quality shadowsize="4096"/>\n    <global offwidth="1280" offheight="960"/>\n  </visual>\n  <default>', 1)
    # 1) 기본 클래스: 휠체어(충돌 비트 2), 벽·장애물(4)
    src = src.replace('    <default class="piper">',
                      '    <default class="chair">\n      <geom contype="2" conaffinity="2"/>\n    </default>\n'
                      '    <default class="wall">\n      <geom contype="4" conaffinity="2" rgba=".90 .88 .83 1"/>\n    </default>\n'
                      '    <default class="decor">\n      <geom contype="0" conaffinity="0" mass="0"/>\n    </default>\n'
                      '    <default class="piper">', 1)
    # 2) 누름 스타일러스 + press_tip (스타일러스 끝) + touch
    src = src.replace('<site name="ee_site"',
                      '<geom name="stylus" type="capsule" fromto="0 0 .094 0 0 .105" size=".004" rgba=".85 .85 .88 1" '
                      'contype="1" conaffinity="1" friction="0.8 0.005 0.0001"/>\n'
                      '                      <site name="press_tip" pos="0 0 .109" size=".004" rgba="1 0 1 .9"/>\n'
                      '                      <site name="touch_tip" type="sphere" pos="0 0 .105" size=".007" rgba="0 0 0 0"/>\n'
                      '                      <site name="ee_site"', 1)
    # 3) 주행 베이스 + 라이다
    lidar = ""
    for k in range(N_LIDAR):
        a = 2 * math.pi * k / N_LIDAR
        lidar += (f'      <site name="lidar_{k:02d}" pos="{LIDAR_POS[0]} {LIDAR_POS[1]} {LIDAR_POS[2]}" size=".005" '
                  f'zaxis="{math.cos(a):.5f} {math.sin(a):.5f} 0" rgba="1 0 0 .3" group="4"/>\n')
    base = (f'    <body name="wheelchair_base" pos="0 0 0" childclass="chair">\n'
            f'      <inertial pos=".02 0 .40" mass="120" diaginertia="7 9 11"/>\n'
            f'      <joint name="base_x" type="slide" axis="1 0 0" damping="5"/>\n'
            f'      <joint name="base_y" type="slide" axis="0 1 0" damping="5"/>\n'
            f'      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="2"/>\n'
            f'      <site name="base_site" pos="0 0 .05" size=".02" rgba=".1 .5 1 .6"/>\n'
            f'{lidar}')
    src = src.replace('    <body name="wheelchair_base" pos="0 0 0">\n', base, 1)
    # 4) 원본 패널·테이블·큐브·원래 조명/바닥 제거
    for name in ("elevator_panel", "side_table", "object_cube"):
        src = re.sub(rf'    <body name="{name}".*?\n    </body>\n', "", src, flags=re.S)
    src = re.sub(r'    <light name="keylight".*?/>\n', "", src)
    src = re.sub(r'    <light name="fill".*?/>\n', "", src)
    src = re.sub(r'    <geom name="floor" type="plane"[^\n]*\n',
                 '    <geom name="floor" type="plane" pos="3.5 1.8 0" size="4.5 2.5 .1" material="floor_tile"/>\n', src)
    # 5) 우리 집 현관문 + 도어락
    lx, ly, lz = LOCK_REL_NOMINAL
    keys = ""
    for row, kz in zip(KEY_ROWS, KEY_ZS):
        for k, kx in zip(row, KEY_XS):
            n = KEY_NAME[k]
            keys += f'''          <body name="key_{n}" pos="{kx} {HOUSING_FRONT + KEY_PROTRUDE - 0.004} {kz}">
            <joint name="key_{n}_slide" type="slide" axis="0 -1 0" range="0 {KEY_TRAVEL}" limited="true" stiffness="180" damping="0.4"/>
            <geom name="key_{n}_geom" type="box" size="{KEY_HALF} .004 {KEY_HALF}" rgba="{KEY_RGBA}" mass=".004" contype="1" conaffinity="1"/>
            <geom name="key_{n}_label" type="box" size="{KEY_HALF - 0.0006} .0004 {KEY_HALF - 0.0006}" pos="0 .0042 0" material="key_{n}" class="decor"/>
            <site name="key_{n}_target" pos="0 .02 0" size=".004" rgba="1 .85 .1 .8"/>
          </body>
'''
    hx, hy, hz = HOUSING_HALF
    home = f'''    <body name="home_door" pos="{DOOR_NOMINAL_X} 0 0" childclass="decor">
{door_geoms("home", UNIT)}      <body name="doorlock" pos="{lx} {ly} {lz}">
        <geom name="lock_housing" type="box" pos="0 {hy} 0" size="{hx} {hy} {hz}" rgba=".16 .17 .18 1" contype="1" conaffinity="1"/>
        <geom name="lock_glass" type="box" pos="0 {2 * hy + 0.0002} .01" size=".034 .0002 .054" rgba=".05 .06 .07 1"/>
        <geom name="lock_led" type="box" pos="0 {2 * hy + 0.0006} {LED_Z}" size=".012 .0006 .0035" rgba="{' '.join(map(str, LED_IDLE))}"/>
        <geom name="lock_brand" type="box" pos="0 {2 * hy + 0.0003} -.072" size=".018 .0003 .004" rgba=".55 .56 .58 1"/>
        <geom name="lock_handle_rose" type="cylinder" pos="0 .012 -.21" quat=".7071068 .7071068 0 0" size=".026 .012" rgba=".70 .68 .64 1"/>
        <geom name="lock_handle_lever" type="box" pos=".055 .034 -.21" size=".065 .012 .011" rgba=".74 .72 .68 1"/>
{keys}      </body>
    </body>
'''
    # 이웃집 문
    nb = ""
    for unit, side, x in NEIGHBORS:
        if side == "near":
            nb += f'    <body name="door_{unit}" pos="{x} 0 0" childclass="decor">\n{door_geoms("d" + unit, unit, 1)}{neighbor_lock("d" + unit, 1)}    </body>\n'
        else:
            nb += f'    <body name="door_{unit}" pos="{x} 3.6 0" childclass="decor">\n{door_geoms("d" + unit, unit, -1)}{neighbor_lock("d" + unit, -1)}    </body>\n'
    # 장애물: 택배(상자 + 테이프 띠 + 송장) / 사람(원기둥 + 머리 + 어깨)
    obst = ""
    for i, (kind, size, rgba) in enumerate(OBSTACLES):
        s = [float(v) for v in size.split()]
        h = s[-1]
        if kind == "parcel":
            extra = (f'      <geom name="obs_{i}_tape" type="box" size="{s[0] + .001} .03 {s[2] + .001}" rgba=".78 .70 .52 1" class="decor"/>\n'
                     f'      <geom name="obs_{i}_label" type="box" pos="{s[0] * .3} {-s[1] - .001} {s[2] * .4}" size=".06 .001 .04" rgba=".96 .96 .94 1" class="decor"/>\n')
            g = f'      <geom name="obs_{i}_geom" class="wall" type="box" size="{size}" rgba="{rgba}"/>\n'
        else:
            r = s[0]
            extra = (f'      <geom name="obs_{i}_head" type="sphere" pos="0 0 {h + .11}" size=".10" rgba=".86 .70 .58 1" class="decor"/>\n'
                     f'      <geom name="obs_{i}_hair" type="sphere" pos="0 .01 {h + .14}" size=".095" rgba=".12 .10 .09 1" class="decor"/>\n'
                     f'      <geom name="obs_{i}_shoulder" type="capsule" fromto="{-r - .02} 0 {h - .12} {r + .02} 0 {h - .12}" size=".06" rgba="{rgba}" class="decor"/>\n'
                     f'      <geom name="obs_{i}_legs" type="cylinder" pos="0 0 {-h + .40}" size="{r * .92} .40" rgba=".18 .18 .22 1" class="decor"/>\n')
            g = f'      <geom name="obs_{i}_geom" class="wall" type="cylinder" size="{size}" rgba="{rgba}"/>\n'
        obst += f'    <body name="obs_{i}" pos="{PARK[0] + 2 * i} {PARK[1]} {h}">\n{g}{extra}    </body>\n'
    # 복도 꾸밈
    decor = ""
    for i, x in enumerate([0.1 + 0.6 * k for k in range(13)]):
        decor += f'    <geom name="tile_seam_x_{i}" type="box" pos="{x:.2f} 1.8 .0015" size=".004 1.8 .0015" rgba=".60 .58 .54 1" class="decor"/>\n'
    for i, y in enumerate([0.6 * k for k in range(1, 6)]):
        decor += f'    <geom name="tile_seam_y_{i}" type="box" pos="3.5 {y:.2f} .0015" size="4.0 .004 .0015" rgba=".60 .58 .54 1" class="decor"/>\n'
    for i, x in enumerate((1.0, 3.2, 5.4)):
        decor += (f'    <geom name="ceil_light_{i}" type="box" pos="{x} 1.8 2.535" size=".30 .30 .01" material="glow_light" '
                  f'class="decor" group="3"/>\n'
                  f'    <light name="hall_{i}" pos="{x} 1.8 2.45" dir="0 0 -1" diffuse=".42 .41 .38" specular=".1 .1 .1" '
                  f'attenuation="1 .05 0" castshadow="true"/>\n')
    hall = f'''    <light name="window_light" pos="7.3 1.8 1.8" dir="-1 0 -.3" diffuse=".25 .27 .30" castshadow="false"/>
    <light name="ambient_fill" pos="3.5 1.8 4" dir="0 0 -1" diffuse=".18 .18 .18" castshadow="false" directional="true"/>
    <geom name="wall_panel" class="wall" type="box" pos="3.5 -.05 1.29" size="4.0 .05 1.29" material="wall_paint"/>
    <geom name="wall_far" class="wall" type="box" pos="3.5 3.65 1.29" size="4.0 .05 1.29" material="wall_paint"/>
    <geom name="wall_left" class="wall" type="box" pos="-.55 1.8 1.29" size=".05 1.9 1.29" material="wall_paint"/>
    <geom name="wall_right" class="wall" type="box" pos="7.55 1.8 1.29" size=".05 1.9 1.29" material="wall_paint"/>
    <geom name="ceiling" type="box" pos="3.5 1.8 2.58" size="4.05 1.85 .04" rgba=".93 .92 .89 1" class="decor" group="3"/>
    <geom name="wainscot_far" type="box" pos="3.5 3.597 .45" size="4.0 .003 .45" rgba=".80 .76 .68 1" class="decor"/>
    <geom name="chair_rail_far" type="box" pos="3.5 3.592 .91" size="4.0 .008 .012" rgba=".58 .50 .40 1" class="decor"/>
    <geom name="baseboard_near" type="box" pos="3.5 .01 .05" size="4.0 .01 .05" rgba=".42 .36 .30 1" class="decor"/>
    <geom name="baseboard_far" type="box" pos="3.5 3.59 .05" size="4.0 .01 .05" rgba=".42 .36 .30 1" class="decor"/>
    <geom name="window" type="box" pos="7.497 1.8 1.55" size=".003 .8 .55" material="glow_window" class="decor"/>
    <geom name="window_frame_h" type="box" pos="7.494 1.8 1.55" size=".004 .82 .02" rgba=".85 .85 .85 1" class="decor"/>
    <geom name="window_frame_v" type="box" pos="7.494 1.8 1.55" size=".004 .02 .57" rgba=".85 .85 .85 1" class="decor"/>
    <geom name="fire_cabinet" type="box" pos="4.5 3.58 1.15" size=".30 .02 .38" rgba=".78 .12 .10 1" class="decor"/>
    <geom name="fire_sign" type="box" pos="4.5 3.559 1.38" size=".12 .001 .05" material="fire_sign" class="decor"/>
    <geom name="exit_sign" type="box" pos="7.44 1.8 2.25" size=".01 .16 .06" material="glow_exit" class="decor"/>
{decor}{home}{nb}{obst}'''
    src = src.replace('  </worldbody>', hall + '  </worldbody>', 1)
    # 6) 재질·텍스처
    mats = "".join(f'    <texture name="key_{KEY_NAME[k]}" type="2d" file="key_{KEY_NAME[k]}.png"/>\n'
                   f'    <material name="key_{KEY_NAME[k]}" texture="key_{KEY_NAME[k]}" emission=".15"/>\n' for k in KEYS)
    mats += "".join(f'    <texture name="unit_{u}" type="2d" file="unit_{u}.png"/>\n'
                    f'    <material name="unit_{u}" texture="unit_{u}"/>\n' for u in [UNIT] + [n[0] for n in NEIGHBORS])
    mats += ('    <texture name="fire_sign" type="2d" file="fire.png"/>\n    <material name="fire_sign" texture="fire_sign"/>\n'
             '    <texture name="floor_tile" type="2d" builtin="checker" rgb1=".74 .72 .67" rgb2=".70 .68 .63" width="512" height="512"/>\n'
             '    <material name="floor_tile" texture="floor_tile" texrepeat="12 6" reflectance=".02"/>\n'
             '    <material name="wall_paint" rgba=".92 .90 .85 1"/>\n'
             '    <material name="glow_light" rgba="1 .98 .92 1" emission="1"/>\n'
             '    <material name="glow_window" rgba=".70 .82 .95 1" emission=".6"/>\n'
             '    <material name="glow_exit" rgba=".10 .70 .30 1" emission=".8"/>\n')
    src = src.replace('  </asset>', mats + '  </asset>', 1)
    # 7) 액추에이터 (팔 7개 뒤에 붙임) + 센서
    src = src.replace('  </actuator>', '''    <velocity name="base_vx" joint="base_x" kv="3000" ctrlrange="-.6 .6" forcerange="-2500 2500"/>
    <velocity name="base_vy" joint="base_y" kv="3000" ctrlrange="-.6 .6" forcerange="-2500 2500"/>
    <velocity name="base_wz" joint="base_yaw" kv="400" ctrlrange="-1.2 1.2" forcerange="-600 600"/>
  </actuator>''', 1)
    sens = "".join(f'    <rangefinder name="lidar_{k:02d}" site="lidar_{k:02d}"/>\n' for k in range(N_LIDAR))
    sens += "".join(f'    <jointpos name="key_{KEY_NAME[k]}_pos" joint="key_{KEY_NAME[k]}_slide"/>\n' for k in KEYS)
    sens += '    <touch name="touch_tip" site="touch_tip"/>\n'
    # 키와 하우징은 겹쳐 있음(키가 하우징 안으로 들어감). 정적 몸체라 부모-자식 충돌 제외가 안 걸려 직접 제외
    excl = "".join(f'    <exclude body1="doorlock" body2="key_{KEY_NAME[k]}"/>\n' for k in KEYS)
    src = src.replace('  <equality>', f'  <contact>\n{excl}  </contact>\n  <equality>', 1)
    src = src.replace('</mujoco>', f'  <sensor>\n{sens}  </sensor>\n</mujoco>')
    src = src.replace('Educational low-poly assembly', 'kw-vip scene_v4 (models/make_scene_v4.py): mobile base + apartment corridor + door lock keypad.\n  Educational low-poly assembly', 1)
    return src


if __name__ == "__main__":
    make_textures()
    DST.write_text(build())
    print("written", DST)
