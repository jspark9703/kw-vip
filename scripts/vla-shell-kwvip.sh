#!/bin/bash
# vla-shell 과 같은 컨테이너 옵션 + kw-vip 저장소 마운트.  vipuser 로 실행한다.
#
#   scripts/vla-shell-kwvip.sh                       # 대화형 셸 (저장소 루트에서 시작)
#   scripts/vla-shell-kwvip.sh python run_demo.py "2층 눌러줘"
#   scripts/vla-shell-kwvip.sh python eval/run_eval.py --runner ik_gt --parser rules
#
# 원래 vla-shell 과 다른 점
#  1) /srv/vla/workspace 만 보이는 문제 -> 저장소($REPO)를 같은 경로로 추가 마운트 (읽기 전용 아님: 코드 수정은 jspark 가 함)
#  2) 결과는 vipuser 가 쓸 수 있는 /results/kw-vip (= 호스트 /srv/vla/results/kw-vip) 에 저장 (KWVIP_OUT)
#  3) 이미지 선택: $KWVIP_IMAGE > vla-lab:x86(있으면) > kw-vip-lab:x86.
#     이미지가 지워졌으면(서버의 Coolify DockerCleanupJob 이 매일 00:00 에 미사용 이미지를 정리함)
#     백업 tar 에서 docker load, 그것도 없으면 docker/Dockerfile 로 다시 빌드
#  4) kw-vip-lab 이미지는 파이썬 패키지를 저장소의 .venv 에서 가져온다 (호스트와 같은 Python 3.12)
#  5) --network host : 호스트에서 도는 Ollama(127.0.0.1:11434)에 접근
set -euo pipefail

# 저장소 경로: vipuser 는 호스트에서 /home/jspark 를 통과할 수 없을 수 있으므로(권한 750) 경로를 직접 받는다.
# 바인드 마운트는 docker 데몬(root)이 하므로 호스트 권한과 무관하게 컨테이너 안에서는 보인다.
REPO="${KWVIP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)}"
RESULTS_HOST="${KWVIP_RESULTS_HOST:-/srv/vla/results}"
IMAGE_TAR="${KWVIP_IMAGE_TAR:-$RESULTS_HOST/kw-vip/images/kw-vip-lab_x86.tar}"

have_image() { docker image inspect "$1" >/dev/null 2>&1; }

if [ -n "${KWVIP_IMAGE:-}" ]; then
    IMAGE="$KWVIP_IMAGE"
elif have_image vla-lab:x86; then
    IMAGE=vla-lab:x86
else
    IMAGE=kw-vip-lab:x86
fi

if ! have_image "$IMAGE"; then
    if [ "$IMAGE" = "kw-vip-lab:x86" ] && [ -f "$IMAGE_TAR" ]; then
        echo "[kw-vip] $IMAGE 이 없어 백업에서 복구: $IMAGE_TAR" >&2
        docker load -i "$IMAGE_TAR" >&2
    elif [ "$IMAGE" = "kw-vip-lab:x86" ]; then
        echo "[kw-vip] $IMAGE 이 없어 docker/Dockerfile 로 빌드합니다" >&2
        if [ -r "$REPO/docker/Dockerfile" ]; then docker build -t "$IMAGE" "$REPO/docker" >&2
        else echo "[kw-vip] Dockerfile 을 읽을 수 없습니다. jspark 계정에서: docker build -t $IMAGE $REPO/docker" >&2; exit 1; fi
    else
        echo "[kw-vip] 이미지 $IMAGE 이 없습니다" >&2; exit 1
    fi
fi
# 미사용 이미지 정리(prune)로 지워지지 않도록 이 이미지를 참조하는 정지 컨테이너를 하나 유지
if [ "$IMAGE" = "kw-vip-lab:x86" ] && ! docker container inspect kwvip-lab-keep >/dev/null 2>&1; then
    docker create --name kwvip-lab-keep "$IMAGE" true >/dev/null 2>&1 || true
fi

mkdir -p "$RESULTS_HOST/kw-vip" 2>/dev/null || true

ENV_ARGS=(-e HOME=/workspace -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl
          -e HF_HOME=/cache/huggingface -e TORCH_HOME=/cache/torch
          -e KWVIP_OUT=/results/kw-vip -e NVIDIA_DRIVER_CAPABILITIES=all
          -e "OLLAMA_HOST=${OLLAMA_HOST:-http://127.0.0.1:11434}")
[ -n "${KWVIP_LLM_MODEL:-}" ] && ENV_ARGS+=(-e "KWVIP_LLM_MODEL=$KWVIP_LLM_MODEL")
if [ "$IMAGE" = "kw-vip-lab:x86" ]; then
    if [ -r "$REPO" ] && [ ! -x "$REPO/.venv/bin/python" ]; then
        echo "[kw-vip] $REPO/.venv 가 없습니다 (jspark 계정에서 scripts/setup_venv.sh 실행 필요)" >&2; exit 1
    fi
    ENV_ARGS+=(-e "PATH=$REPO/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" -e "VIRTUAL_ENV=$REPO/.venv")
fi

TTY_ARGS=(-i)
[ -t 0 ] && [ -t 1 ] && TTY_ARGS=(-it)

echo "[kw-vip] image=$IMAGE repo=$REPO results=$RESULTS_HOST/kw-vip" >&2
exec docker run --rm "${TTY_ARGS[@]}" \
    --gpus all --ipc=host --shm-size=16g --network host \
    --user "$(id -u):$(id -g)" \
    "${ENV_ARGS[@]}" \
    -v /srv/vla/workspace:/workspace \
    -v /srv/vla/cache:/cache \
    -v "$RESULTS_HOST":/results \
    -v "$REPO":"$REPO" \
    -w "$REPO" \
    "$IMAGE" "${@:-/bin/bash}"
