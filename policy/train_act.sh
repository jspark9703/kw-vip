#!/bin/bash
# LeRobot(0.7.0) ACT 학습.  vla-shell-kwvip.sh 안에서 실행 (결과는 /results/kw-vip/train/<name>)
#   bash policy/train_act.sh press_v1_small 3000 8     # 과적합 확인용
#   bash policy/train_act.sh press_v1 30000 32         # 본 학습
# 사전학습 가중치: ACT 기본 설정의 ResNet18 백본(torchvision ImageNet) -> 발표에 출처 명시
set -euo pipefail
DS=${1:-press_v1_small}; STEPS=${2:-3000}; BS=${3:-8}
OUT_ROOT=${KWVIP_OUT:-outputs}
NAME=${NAME:-act_${DS}_s${STEPS}}
rm -rf "$OUT_ROOT/train/$NAME"
lerobot-train \
  --dataset.repo_id=kwvip/$DS --dataset.root="$OUT_ROOT/lerobot/$DS" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=$BS --steps=$STEPS --save_freq=$(( STEPS / 3 )) --log_freq=200 \
  --num_workers=4 --trackio.enable=false --job_name=$NAME \
  --output_dir="$OUT_ROOT/train/$NAME" "${@:4}"
du -sh "$OUT_ROOT/train/$NAME/checkpoints/last/pretrained_model"
