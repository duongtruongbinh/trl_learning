# Vision-Language DPO Playground

This repo fine-tunes Qwen3-VL models with Direct Preference Optimization (DPO), starting from raw VQA-style datasets on disk and ending with quantitative VQA accuracy reports for both the base and fine-tuned checkpoints.

## Repo layout

```
data/
  convert_vqa_to_qwen3vl.py   # raw JSONL -> preference format
  preprocessing.py            # shared dataset helpers
models/
  qwen.py                     # text-only Qwen helpers (legacy)
  weights/                    # local checkpoints (ignored)
scripts/
  train_vl_dpo.sh             # single-run training entrypoint
  inference_vl.sh             # run baseline + finetuned generations
  evaluate_vqa.sh             # score predictions via VQA accuracy
  run_full_evaluation.sh      # end-to-end inference + scoring
src/
  train_vl.py                 # Qwen3-VL DPO trainer
  inference_vl.py             # multimodal inference loop
  evaluate_vqa.py             # VQA accuracy metric
```

`results/`, `models/weights/`, and `data/data_raw/` stay local (see `.gitignore`).

## Setup

```bash
pip install -r requirements.txt
accelerate config
```

Add `tqdm`, `trl`, `transformers==4.45+`, and CUDA-capable PyTorch; requirements already target these versions.

## Data preparation

1. Place raw VQA JSONL files (e.g., VQAv2/TextVQA/OKVQA) under `/mnt/VLAI_data/...`.
2. Convert them into a DPO-ready preference set with image paths:
   ```bash
   python data/convert_vqa_to_qwen3vl.py
   ```
   Outputs land in `data/data_raw/*_dpo.json` and are loaded automatically by `train_vl.py`.

## Training

Update `scripts/train_vl_dpo.sh` with the desired dataset/model paths, then launch:

```bash
bash scripts/train_vl_dpo.sh
```

Key defaults:
- Base model: `/mnt/dataset1/pretrained_fm/Qwen_Qwen3-VL-2B-Instruct`
- Dataset: `data/data_raw/okvqa_train_dpo.json`
- LoRA-enabled DPO with reference model (`--use_ref_model`)

Adjust batch size, epochs, and dataset path through CLI flags in the script or `src/train_vl.py`.

## Inference & evaluation

1. **Inference only**
   ```bash
   bash scripts/inference_vl.sh
   ```
   Produces `results/predictions_{baseline,finetuned}_okvqa.json`.

2. **Evaluation only**
   ```bash
   bash scripts/evaluate_vqa.sh
   ```
   Loads the prediction files above and computes VQA accuracy via `src/evaluate_vqa.py`.

3. **Full pipeline**
   ```bash
   bash scripts/run_full_evaluation.sh
   ```
   Runs baseline+finetuned inference and prints a summary comparison.

`src/inference_vl.py` handles multimodal prompts (image + text) while `src/evaluate_vqa.py` normalizes answers and applies the standard VQA scoring rule.

