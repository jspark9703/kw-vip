# 사용법: python models/make_scene_v3.py [models/orig/wheelchair_piper_lite.xml] [models/scene_v3.xml]
"""원본 XML -> scene_v3.xml : 주행 휠체어 + 엘리베이터 홀 + 숫자 버튼 6개(2x3) + 장애물 풀 + 라이다.

변경점 (발표에서 설명할 것)
 1. 주행: wheelchair_base 에 base_x / base_y (slide, 월드 축) + base_yaw (hinge z) 관절, 속도 액추에이터 3개.
    제어기가 (v, ω) -> (v cosθ, v sinθ, ω) 로 넣어 차동구동(옆 미끄러짐 없음) 제약을 건다. z 는 고정(바닥 접촉 없음).
    질량 120kg 근사 관성. 액추에이터 순서는 팔 7개 다음 -> ctrl[0:7] 의미는 v2 와 같다.
 2. 충돌 비트: 바닥 1 / 휠체어 2 / 벽·장애물 4.  휠체어는 바닥과 충돌하지 않고 벽·장애물·패널과 충돌(충돌 수 집계).
 3. 엘리베이터 홀: 패널이 붙은 벽(y=0), 맞은편 벽(y=3.6), 양 끝 벽, 엘리베이터 문(시각용).
 4. 패널 2x3: 버튼 6개 모두 같은 색, 앞면 라벨 텍스처(숫자 1~6)로만 구분. 격자 칸 배치는 sim/world.py 가 시드로 섞는다.
    누름이 등록되면(sim 에서) 버튼 색을 램프 색으로 바꾼다.
 5. 센서: 라이다(rangefinder 72빔, 5° 간격, 높이 0.45m), 버튼 jointpos 6개, 손가락 패드 touch 2개.
 6. 장애물 풀 obs_0..5 (박스·원기둥, 높이 0.8~0.9m). 쓰지 않는 것은 장면 밖(x=20)에 둔다.
 7. 원본의 테이블·큐브는 제거(주행 경로와 겹침).
"""
import math
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "orig" / "wheelchair_piper_lite.xml"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "scene_v3.xml"
TEX_DIR = ROOT / "assets" / "textures"
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"

# ---- 배치 상수 (sim/world.py 와 공유: world.py 가 이 모듈을 import) ----
PANEL_NOMINAL = (5.0, 0.035, 1.09)      # 패널 body 중심 (벽 앞면 y=0)
DOCK_REL = (0.37, -0.87, 1.09)          # v2 에서 검증된 '휠체어 기준 패널 위치' -> 도킹 자세 계산에 사용
START_NOMINAL = (0.8, 1.8, 0.0)         # 출발 (x, y, yaw)
SLOTS = [(-0.07, 0.035), (0.0, 0.035), (0.07, 0.035), (-0.07, -0.035), (0.0, -0.035), (0.07, -0.035)]  # (x, z)
DEFAULT_LAYOUT = (4, 5, 6, 1, 2, 3)     # 칸 순서별 층 (위줄 4 5 6 / 아래줄 1 2 3)
BUTTON_RGBA = ".86 .86 .82 1"
LAMP_RGBA = (1.0, 0.62, 0.12, 1.0)
N_LIDAR = 72
LIDAR_POS = (0.06, -0.03, 0.45)
OBSTACLES = [  # (type, size, rgba)
    ("box", ".22 .22 .45", ".55 .42 .30 1"),
    ("cylinder", ".18 .40", ".25 .45 .30 1"),
    ("box", ".28 .25 .42", ".40 .40 .55 1"),
    ("cylinder", ".22 .45", ".60 .35 .25 1"),
    ("box", ".15 .35 .40", ".50 .50 .50 1"),
    ("cylinder", ".15 .42", ".30 .30 .65 1"),
]
PARK = (20.0, 20.0)


def make_digit_textures():
    """버튼 라벨: 밝은 회색 바탕 + 검은 숫자 (128x128). 같은 폰트로 인식 템플릿도 만든다(perception/digits.py)."""
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(FONT), 104)
    bg = tuple(int(float(v) * 255) for v in BUTTON_RGBA.split()[:3])
    for n in range(1, 7):
        im = Image.new("RGB", (128, 128), bg)
        dr = ImageDraw.Draw(im)
        l, t, r, b = dr.textbbox((0, 0), str(n), font=font)
        dr.text(((128 - (r - l)) / 2 - l, (128 - (b - t)) / 2 - t), str(n), fill=(15, 15, 15), font=font)
        # MuJoCo 박스 +y 면에 2d 텍스처가 좌우 반전되어 입혀지므로 미리 뒤집어 저장 (렌더링에서 정상으로 보임)
        im.transpose(Image.FLIP_LEFT_RIGHT).save(TEX_DIR / f"digit_{n}.png")


def build() -> str:
    src = SRC.read_text()
    # 0) 텍스처 경로
    src = src.replace('<compiler angle="radian" autolimits="true"/>',
                      '<compiler angle="radian" autolimits="true" texturedir="../assets/textures"/>')
    src = src.replace('<mujoco model="wheelchair_piper_lite">', '<mujoco model="wheelchair_piper_v3">')
    # 장면 범위 고정: 장면 밖에 둔 장애물 때문에 extent 가 커지면 카메라 near plane(=znear*extent)이 수십 cm 가 되어
    # 손목 카메라가 가까운 패널을 잘라낸다 -> extent 5m, znear 0.002 (=1cm)
    src = src.replace('  <default>', '  <statistic extent="5" center="3.5 1.8 1"/>\n  <visual>\n    <map znear="0.002"/>\n  </visual>\n  <default>', 1)
    # 1) 휠체어 기본 클래스 (충돌 비트 2)
    src = src.replace('    <default class="piper">',
                      '    <default class="chair">\n      <geom contype="2" conaffinity="2"/>\n    </default>\n'
                      '    <default class="wall">\n      <geom contype="4" conaffinity="2" rgba=".86 .84 .78 1"/>\n    </default>\n'
                      '    <default class="piper">', 1)
    # 2) press_tip (v2 와 같음)
    src = src.replace('<site name="ee_site"', '<site name="press_tip" pos="0 0 .094" size=".008" rgba="1 0 1 .9"/>\n'
                      '                      <site name="ee_site"', 1)
    # 손가락 패드 touch site
    src = src.replace('<geom name="finger_pad_left" type="box" size=".012 .014 .005" pos="0 -.055 -.004"',
                      '<site name="touch_left" type="box" size=".014 .016 .008" pos="0 -.055 -.004" rgba="0 0 0 0"/>\n'
                      '                        <geom name="finger_pad_left" type="box" size=".012 .014 .005" pos="0 -.055 -.004"')
    src = src.replace('<geom name="finger_pad_right" type="box" size=".012 .014 .005" pos="0 .055 -.004"',
                      '<site name="touch_right" type="box" size=".014 .016 .008" pos="0 .055 -.004" rgba="0 0 0 0"/>\n'
                      '                        <geom name="finger_pad_right" type="box" size=".012 .014 .005" pos="0 .055 -.004"')
    # 3) 주행 베이스
    sx, sy, syaw = START_NOMINAL
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
    # 4) 원본 패널·테이블·큐브 제거
    for name in ("elevator_panel", "side_table", "object_cube"):
        src = re.sub(rf'    <body name="{name}".*?\n    </body>\n', "", src, flags=re.S)
    # 5) 홀·패널·장애물
    px, py, pz = PANEL_NOMINAL
    btns = ""
    for slot, n in zip(SLOTS, DEFAULT_LAYOUT):
        x, z = slot
        btns += f'''      <body name="elevator_button_{n}" pos="{x} .065 {z}">
        <joint name="button_{n}_slide" type="slide" axis="0 -1 0" range="0 .015" limited="true" stiffness="220" damping="3"/>
        <geom name="button_{n}_geom" type="cylinder" size=".022 .015" quat=".7071068 .7071068 0 0" rgba="{BUTTON_RGBA}" mass=".025"/>
        <geom name="button_{n}_label" type="box" size=".0135 .0006 .0135" pos="0 .0157 0" material="digit_{n}" contype="0" conaffinity="0" mass="0"/>
        <site name="button_{n}_target" pos="0 .04 0" size=".010" rgba="1 .85 .1 .8"/>
      </body>
'''
    obst = ""
    for i, (typ, size, rgba) in enumerate(OBSTACLES):
        h = float(size.split()[-1])
        obst += (f'    <body name="obs_{i}" pos="{PARK[0] + 2 * i} {PARK[1]} {h}">\n'
                 f'      <geom name="obs_{i}_geom" class="wall" type="{typ}" size="{size}" rgba="{rgba}"/>\n'
                 f'    </body>\n')
    hall = f'''    <light name="hall_1" pos="1.5 1.8 2.6" dir="0 0 -1" diffuse=".45 .45 .45"/>
    <light name="hall_2" pos="4.5 1.2 2.6" dir="0 -.3 -1" diffuse=".55 .55 .55"/>
    <geom name="wall_panel" class="wall" type="box" pos="3.5 -.05 1.25" size="4.0 .05 1.25"/>
    <geom name="wall_far" class="wall" type="box" pos="3.5 3.65 1.25" size="4.0 .05 1.25"/>
    <geom name="wall_left" class="wall" type="box" pos="-.55 1.8 1.25" size=".05 1.9 1.25"/>
    <geom name="wall_right" class="wall" type="box" pos="7.55 1.8 1.25" size=".05 1.9 1.25"/>
    <geom name="elevator_door" type="box" pos="{px + 0.80} .004 1.05" size=".50 .004 1.05" rgba=".62 .66 .70 1" contype="0" conaffinity="0"/>
    <geom name="elevator_frame" type="box" pos="{px + 0.80} .002 1.07" size=".56 .002 1.10" rgba=".35 .37 .40 1" contype="0" conaffinity="0"/>
    <body name="elevator_panel" pos="{px} {py} {pz}">
      <geom type="box" name="panel" size=".13 .035 .10" rgba=".30 .33 .37 1" contype="1" conaffinity="3"/>
{btns}    </body>
{obst}'''
    src = src.replace('  </worldbody>', hall + '  </worldbody>', 1)
    # 바닥: 홀 전체를 덮도록
    src = src.replace('<geom name="floor" type="plane" pos="0 0 0" size="5 5 .1"',
                      '<geom name="floor" type="plane" pos="3.5 1.8 0" size="4.5 2.5 .1"')
    # 6) 재질·텍스처
    mats = "".join(f'    <texture name="digit_{n}" type="2d" file="digit_{n}.png"/>\n'
                   f'    <material name="digit_{n}" texture="digit_{n}"/>\n' for n in range(1, 7))
    src = src.replace('  </asset>', mats + '  </asset>', 1)
    # 7) 액추에이터 (팔 7개 뒤에 붙임) + 센서
    src = src.replace('  </actuator>', '''    <velocity name="base_vx" joint="base_x" kv="3000" ctrlrange="-.6 .6" forcerange="-2500 2500"/>
    <velocity name="base_vy" joint="base_y" kv="3000" ctrlrange="-.6 .6" forcerange="-2500 2500"/>
    <velocity name="base_wz" joint="base_yaw" kv="400" ctrlrange="-1.2 1.2" forcerange="-600 600"/>
  </actuator>''', 1)
    sens = "".join(f'    <rangefinder name="lidar_{k:02d}" site="lidar_{k:02d}"/>\n' for k in range(N_LIDAR))
    sens += "".join(f'    <jointpos name="button_{n}_pos" joint="button_{n}_slide"/>\n' for n in range(1, 7))
    sens += '    <touch name="touch_left" site="touch_left"/>\n    <touch name="touch_right" site="touch_right"/>\n'
    src = src.replace('</mujoco>', f'  <sensor>\n{sens}  </sensor>\n</mujoco>')
    src = src.replace('Educational low-poly assembly', 'kw-vip scene_v3 (models/make_scene_v3.py): mobile base + elevator hall.\n  Educational low-poly assembly', 1)
    return src


if __name__ == "__main__":
    make_digit_textures()
    DST.write_text(build())
    print("written", DST)
