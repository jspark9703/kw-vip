# 사용법: python make_scene_v2.py wheelchair_piper_lite.xml scene_v2.xml
"""원본 XML → scene_v2.xml : 버튼 3개(가로 1줄), press_tip site, 큐브 축소·이동, 목표 영역 site."""
import re, sys
src = open(sys.argv[1]).read()
# 1) press_tip site (손가락 패드 앞면)
src = src.replace('<site name="ee_site"', '<site name="press_tip" pos="0 0 .094" size=".008" rgba="1 0 1 .9"/>\n                      <site name="ee_site"', 1)
# 2) 패널 + 버튼 3개
colors = {1: ".95 .35 .35 1", 2: ".35 .85 .40 1", 3: ".29 .67 .98 1"}
btns = ""
for i, x in zip((1, 2, 3), (-.08, 0, .08)):
    btns += f'''      <body name="elevator_button_{i}" pos="{x} .065 0">
        <joint name="button_{i}_slide" type="slide" axis="0 -1 0" range="0 .015" limited="true" stiffness="220" damping="3"/>
        <geom name="button_{i}_geom" type="cylinder" size=".025 .015" quat=".7071068 .7071068 0 0" rgba="{colors[i]}" mass=".025"/>
        <site name="button_{i}_target" pos="0 .04 0" size=".012" rgba="1 .85 .1 .8"/>
      </body>
'''
panel = f'''    <body name="elevator_panel" pos=".37 -.87 1.09">
      <geom type="box" name="panel" size=".145 .035 .12" rgba=".38 .42 .46 1"/>
{btns}    </body>
'''
src = re.sub(r'    <body name="elevator_panel".*?\n    </body>\n', panel, src, flags=re.S)
# 3) 큐브 5cm, 도달 가능 영역으로 이동 (B 확장용) + 목표 영역
src = src.replace('<body name="object_cube" pos=".60 -.08 .735">', '<body name="object_cube" pos=".45 -.20 .705">')
src = src.replace('size=".035 .035 .035" density="400"', 'size=".025 .025 .025" density="400"')
src = src.replace('<body name="object_cube"', '<site name="place_zone" pos=".45 -.10 .681" size=".06 .06 .001" type="box" rgba=".2 .8 .3 .35"/>\n    <body name="object_cube"', 1)
open(sys.argv[2], "w").write(src)
print("written", sys.argv[2])
