#!/bin/bash
# kw-vip 실행 환경 (GPU + MuJoCo EGL 컨테이너).  jspark 계정에서 저장소 루트 기준으로 실행한다.
#
#   scripts/vla-shell-kwvip.sh                                   # 대화형 셸 (저장소 루트에서 시작)
#   scripts/vla-shell-kwvip.sh python run_demo.py "2층 눌러줘"
#   scripts/vla-shell-kwvip.sh python eval/run_eval.py --runner ik_gt --parser rules
#
# - 이미지: kw-vip-lab:x86 (docker/Dockerfile, Ubuntu 24.04 + EGL). 파이썬 패키지는 저장소의 .venv 를 사용.
#   이미지가 없으면 outputs/images/kw-vip-lab_x86.tar 에서 불러오고, 그것도 없으면 docker/Dockerfile 로 빌드.
#   서버의 미사용 이미지 정리(prune)에 지워지지 않도록 정지 컨테이너 kwvip-lab-keep 이 이미지를 참조한다.
# - 쓰기 위치는 모두 저장소 안: 결과 outputs/ (KWVIP_OUT), 캐시 .cache/ (HF_HOME, TORCH_HOME), HOME .cache/home
# - --network host : 호스트의 Ollama(127.0.0.1:11434)에 접근
set -euo pipefail

REPO="${KWVIP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${KWVIP_IMAGE:-kw-vip-lab:x86}"
IMAGE_TAR="${KWVIP_IMAGE_TAR:-$REPO/outputs/images/kw-vip-lab_x86.tar}"

have_image() { docker image inspect "$1" >/dev/null 2>&1; }

if ! have_image "$IMAGE"; then
    if [ -f "$IMAGE_TAR" ]; then
        echo "[kw-vip] $IMAGE 이 없어 백업에서 복구: $IMAGE_TAR" >&2
        docker load -i "$IMAGE_TAR" >&2
    else
        echo "[kw-vip] $IMAGE 이 없어 docker/Dockerfile 로 빌드합니다" >&2
        docker build -t "$IMAGE" "$REPO/docker" >&2
    fi
fi
if ! docker container inspect kwvip-lab-keep >/dev/null 2>&1; then
    docker create --name kwvip-lab-keep "$IMAGE" true >/dev/null 2>&1 || true
fi
if [ ! -f "$IMAGE_TAR" ]; then  # 백업이 없으면 한 번 만들어 둔다
    mkdir -p "$(dirname "$IMAGE_TAR")" && docker save "$IMAGE" -o "$IMAGE_TAR" || true
fi

[ -x "$REPO/.venv/bin/python" ] || { echo "[kw-vip] $REPO/.venv 가 없습니다: scripts/setup_venv.sh act 먼저 실행" >&2; exit 1; }
mkdir -p "$REPO/outputs" "$REPO/.cache/home" "$REPO/.cache/huggingface" "$REPO/.cache/torch"

TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)

echo "[kw-vip] image=$IMAGE repo=$REPO outputs=$REPO/outputs" >&2
exec docker run --rm "${TTY_ARGS[@]}" \
    --gpus all --ipc=host --shm-size=16g --network host \
    --user "$(id -u):$(id -g)" \
    -e HOME="$REPO/.cache/home" -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    -e HF_HOME="$REPO/.cache/huggingface" -e TORCH_HOME="$REPO/.cache/torch" \
    -e KWVIP_OUT="$REPO/outputs" -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e "OLLAMA_HOST=${OLLAMA_HOST:-http://127.0.0.1:11434}" \
    -e "KWVIP_LLM_MODEL=${KWVIP_LLM_MODEL:-qwen2.5:3b}" \
    -e "PATH=$REPO/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e "VIRTUAL_ENV=$REPO/.venv" \
    -v "$REPO":"$REPO" \
    -w "$REPO" \
    "$IMAGE" "${@:-/bin/bash}"
