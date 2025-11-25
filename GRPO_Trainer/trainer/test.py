# trainer/eval_grpo.py

import os
import sys
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.gsmk8_dataset import GSMK8Dataset
from trainer.grpo_trainer import evaluate_accuracy


class DummyTrainer:
    """Giả lập interface tối thiểu của GRPOTrainer để dùng lại evaluate_accuracy."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def find_latest_checkpoint(output_dir: str) -> str:
    candidates = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            # checkpoint-500 -> 500
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, path))

    if not candidates:
        print(f"[WARN] Không tìm thấy checkpoint-* trong {output_dir}, "
              f"sẽ dùng trực tiếp thư mục này (có thể lại lỗi config).")
        return output_dir

    candidates.sort(key=lambda x: x[0])
    best_step, best_path = candidates[-1]
    print(f"[INFO] Dùng checkpoint mới nhất: {best_path} (step={best_step})")
    return best_path


def main():
    # ========= 1) Load TEST jsonl via GSMK8Dataset =========
    test_path = os.path.join(ROOT_DIR, "data", "raw", "test.jsonl")
    test_gsm = GSMK8Dataset(test_path)

    test_list = []
    for i in tqdm(range(len(test_gsm)), desc="Converting TEST GSMK8Dataset to HF Dataset"):
        test_list.append(test_gsm[i])

    test_dataset = Dataset.from_list(test_list)

    # ========= 2) Load model + tokenizer từ checkpoint =========
    output_dir = os.path.join(ROOT_DIR, "Qwen2-0.5B-GRPO-math")
    ckpt_dir = find_latest_checkpoint(output_dir)

    base_model_id = "Qwen/Qwen2-0.5B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[INFO] Loading model from: {ckpt_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        torch_dtype=dtype,
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = DummyTrainer(model=model, tokenizer=tokenizer)

    acc = evaluate_accuracy(trainer, eval_dataset=test_dataset, batch_size=4)

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    results_path = os.path.join(log_dir, "test_results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== Qwen2-0.5B-GRPO-math - Test Results ===\n")
        f.write(f"Num samples: {len(test_dataset)}\n")
        f.write(f"Test accuracy: {acc:.4f}\n")

    print(f"\n[INFO] Saved test results to: {results_path}")


if __name__ == "__main__":
    main()
