#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Baseline model inference
echo "Running inference on baseline model..."
CUDA_VISIBLE_DEVICES=0 python src/inference_vl.py \
  --model_name /mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct \
  --dataset_name data/data_raw/okvqa_val_dpo.json \
  --output_file results/predictions_baseline_okvqa.json \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.8 \
  --num_samples 1000

# Fine-tuned model inference
echo "Running inference on fine-tuned model..."
CUDA_VISIBLE_DEVICES=0 python src/inference_vl.py \
  --model_name ./models/weights/Qwen3-VL-2B-DPO-OKVQA \
  --dataset_name data/data_raw/okvqa_val_dpo.json \
  --output_file results/predictions_finetuned_okvqa.json \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.8 \
  --num_samples 1000

echo "Inference completed!"

