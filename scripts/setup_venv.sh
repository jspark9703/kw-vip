#!/bin/bash
# 저장소 루트에 .venv 생성 (jspark 계정에서 1회).  vla-shell-kwvip.sh 의 kw-vip-lab 이미지가 이 venv 를 쓴다.
#   scripts/setup_venv.sh          # 기본 (IK·인식·NLU·평가)
#   scripts/setup_venv.sh act      # + LeRobot/torch (ACT 학습·추론)
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install -U pip
if [ "${1:-}" = "act" ]; then .venv/bin/pip install -r requirements-act.txt; else .venv/bin/pip install -r requirements.txt; fi
chmod -R o+rX .venv   # 컨테이너 안의 vipuser 가 읽을 수 있게
MUJOCO_GL=egl .venv/bin/python -c "import mujoco; r=mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>')); r.close(); print('mujoco', mujoco.__version__, 'EGL OK')"
