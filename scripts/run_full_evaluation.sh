#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Create results directory
mkdir -p results

echo "=========================================="
echo "Full Evaluation Pipeline"
echo "=========================================="
echo ""

# Step 1: Run inference on baseline model
echo "Step 1: Running inference on baseline model..."
CUDA_VISIBLE_DEVICES=0 python src/inference_vl.py \
  --model_name /mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct \
  --dataset_name data/data_raw/okvqa_val_dpo.json \
  --output_file results/predictions_baseline_okvqa.json \
  --max_new_tokens 100 \

echo ""

# Step 2: Run inference on fine-tuned model
echo "Step 2: Running inference on fine-tuned model..."
CUDA_VISIBLE_DEVICES=0 python src/inference_vl.py \
  --model_name ./models/weights/Qwen3-VL-2B-DPO-OKVQA \
  --dataset_name data/data_raw/okvqa_val_dpo.json \
  --output_file results/predictions_finetuned_okvqa.json \
  --max_new_tokens 100 \

echo ""

# Step 3: Evaluate baseline predictions
echo "Step 3: Evaluating baseline model predictions..."
python src/evaluate_vqa.py \
  --predictions_file results/predictions_baseline_okvqa.json \
  --output_file results/evaluation_baseline_okvqa.json

echo ""

# Step 4: Evaluate fine-tuned predictions
echo "Step 4: Evaluating fine-tuned model predictions..."
python src/evaluate_vqa.py \
  --predictions_file results/predictions_finetuned_okvqa.json \
  --output_file results/evaluation_finetuned_okvqa.json

echo ""
echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo ""
echo "Baseline Results:"
python -c "
import json
with open('results/evaluation_baseline_okvqa.json', 'r') as f:
    data = json.load(f)
    print(f\"  Accuracy: {data['overall_accuracy']:.4f} ({data['overall_accuracy']*100:.2f}%)\")
    print(f\"  Correct: {data['correct_predictions']}/{data['total_samples']}\")
"

echo ""
echo "Fine-tuned Results:"
python -c "
import json
with open('results/evaluation_finetuned_okvqa.json', 'r') as f:
    data = json.load(f)
    print(f\"  Accuracy: {data['overall_accuracy']:.4f} ({data['overall_accuracy']*100:.2f}%)\")
    print(f\"  Correct: {data['correct_predictions']}/{data['total_samples']}\")
"

echo ""
echo "All results saved in results/ directory"
echo "=========================================="

