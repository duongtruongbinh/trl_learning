import os
import sys
import re
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.gsmk8_dataset import GSMK8Dataset 

# ==========================
# Helper: number normalization
# ==========================
def normalize_number(ans: str) -> str:
    """
    Normalize numeric answers so that '42', '42.0', '  42 ' become comparable.
    Very simple heuristic: keep only the first integer/float pattern.
    """
    ans = ans.strip().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", ans)
    if m:
        return m.group(0)
    return ans


# ==========================
# Reward function: math accuracy for training
# ==========================
def math_accuracy_reward(completions, solution, **kwargs):
    """
    Reward = 1.0 if model final answer == solution, else 0.0.

    - completions: for chat models, usually:
        [
          [ {"role": "assistant", "content": "..."} ],
          [ {"role": "assistant", "content": "..."} ],
          ...
        ]
    - solution: list[str], from dataset column 'solution'
    """
    # Extract plain text from completions (support both string or chat format)
    completion_texts = []
    for comp in completions:
        if isinstance(comp, list) and comp and isinstance(comp[0], dict):
            completion_texts.append(comp[0].get("content", ""))
        else:
            completion_texts.append(str(comp))

    rewards = []
    for text, sol in zip(completion_texts, solution):
        # Use your dataset's extraction logic to get final answer from model output
        pred = GSMK8Dataset._extract_final_answer(text)
        pred_norm = normalize_number(pred)
        sol_norm = normalize_number(sol)

        if pred_norm and sol_norm and pred_norm == sol_norm:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ==========================
# Evaluation: accuracy on a test HF Dataset
# ==========================
def evaluate_accuracy(trainer: GRPOTrainer, eval_dataset: Dataset, batch_size: int = 4):
    """
    Run the fine-tuned model on eval_dataset and compute accuracy based on final answer.
    """
    model = trainer.model
    tokenizer = trainer.tokenizer  # GRPOTrainer constructs this internally
    model.eval()

    total = 0
    correct = 0

    for start in tqdm(
        range(0, len(eval_dataset), batch_size),
        desc="Evaluating on test set"
    ):
        end = min(start + batch_size, len(eval_dataset))
        batch = eval_dataset[start:end]

        prompts = batch["prompt"]
        solutions = batch["solution"]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # should match/max your prompt length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,   # should align with your completion length
                do_sample=False,      # deterministic eval
                temperature=0.0,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text, sol in zip(texts, solutions):
            pred = GSMK8Dataset._extract_final_answer(text)
            pred_norm = normalize_number(pred)
            sol_norm = normalize_number(sol)

            if pred_norm and sol_norm and pred_norm == sol_norm:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\nTest accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


def main():
    # ========= 1) Load TRAIN jsonl via GSMK8Dataset =========
    train_gsm = GSMK8Dataset("data/train.jsonl")

    train_list = []
    for i in tqdm(range(len(train_gsm)), desc="Converting train GSMK8Dataset to HF Dataset"):
        train_list.append(train_gsm[i])

    train_dataset = Dataset.from_list(train_list)

    # (optional) subset for quick debug
    # train_dataset = train_dataset.select(range(1000))

    # ========= 2) GRPO config =========
    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO-math",
        per_device_train_batch_size=4,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=256,
        logging_steps=10,
        save_steps=500,
        num_train_epochs=1,
        # you can add learning_rate, bf16/fp16, etc.
    )

    # ========= 3) Create GRPOTrainer =========
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=math_accuracy_reward,
        args=training_args,
        train_dataset=train_dataset,
    )

    # ========= 4) Train =========
    trainer.train()

    # ========= 5) Load TEST jsonl & evaluate accuracy =========
    test_gsm = GSMK8Dataset("data/test.jsonl")  # <-- dùng test.jsonl của bạn

    test_list = []
    for i in tqdm(range(len(test_gsm)), desc="Converting test GSMK8Dataset to HF Dataset"):
        test_list.append(test_gsm[i])

    test_dataset = Dataset.from_list(test_list)

    # Evaluate accuracy on test set
    evaluate_accuracy(trainer, test_dataset, batch_size=4)


if __name__ == "__main__":
    main()
