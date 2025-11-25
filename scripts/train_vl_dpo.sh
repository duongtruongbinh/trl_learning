#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

CUDA_VISIBLE_DEVICES=0 accelerate launch src/train_vl.py \
  --model_name /mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct \
  --dataset_name data/data_raw/okvqa_train_dpo.json \
  --dataset_split train \
  --output_dir ./models/weights/Qwen3-VL-2B-DPO-OKVQA \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --max_length 2048 \
  --beta 0.1 \
  --loss_type sigmoid \
  --bf16 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --preprocess_num_proc 16 \
  --dataloader_num_workers 8 \
  --cache_dir ./data/cache \
  --use_ref_model

