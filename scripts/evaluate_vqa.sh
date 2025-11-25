#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Evaluate baseline predictions
echo "Evaluating baseline model predictions..."
python src/evaluate_vqa.py \
  --predictions_file results/predictions_baseline_okvqa.json \
  --output_file results/evaluation_baseline_okvqa.json

# Evaluate fine-tuned predictions
echo "Evaluating fine-tuned model predictions..."
python src/evaluate_vqa.py \
  --predictions_file results/predictions_finetuned_okvqa.json \
  --output_file results/evaluation_finetuned_okvqa.json

echo "Evaluation completed!"

