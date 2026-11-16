# kw-vip 실행용 경량 이미지 (vla-lab:x86 이미지가 서버에서 사라졌을 때의 대체).
#  - 호스트(Ubuntu 24.04)와 같은 Python 3.12 -> 저장소의 .venv 를 그대로 마운트해서 사용
#  - MuJoCo EGL 렌더링에 필요한 GL/EGL 라이브러리만 설치 (NVIDIA EGL 은 nvidia-container-toolkit 이 주입)
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libegl1 libgl1 libglvnd0 libgles2 libopengl0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
LABEL org.kwvip.purpose="kw-vip runtime (keep)"
WORKDIR /workspace
CMD ["/bin/bash"]
