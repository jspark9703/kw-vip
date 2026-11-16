#!/bin/bash
# 실행 환경 점검 (vla-shell-kwvip.sh 안 또는 노트북에서).  문서 '2. vla-shell 환경 점검' 항목.
echo "python: $(which python) $(python -V 2>&1)"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || echo "torch 없음 (ACT 단계에서만 필요)"
python -c "import lerobot; print('lerobot', lerobot.__version__)" 2>/dev/null || echo "lerobot 없음 (ACT 단계에서만 필요)"
MUJOCO_GL=${MUJOCO_GL:-egl} python -c "import mujoco; r=mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>')); r.close(); print('EGL render OK')" 2>/dev/null || echo "EGL 렌더링 실패"
python - <<'PY'
import os, requests
h = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
h = h if h.startswith("http") else "http://" + h
try:
    tags = requests.get(h + "/api/tags", timeout=2).json()
    print("ollama OK:", [m["name"] for m in tags.get("models", [])])
except Exception as e:
    print("ollama 없음/접속 불가 ->", h, "(규칙 파서로 폴백)")
PY
