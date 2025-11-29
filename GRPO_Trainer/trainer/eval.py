import os
import sys

import torch
from datasets import Dataset
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from data.gsmk8_dataset import GSMK8Dataset
from utils.utils import normalize_number
from trainer.train import convert_to_hf_dataset


def evaluate_accuracy(
    trainer,
    eval_dataset: GSMK8Dataset,
    batch_size: int = 4,
    max_prompt_length: int = 128,
    max_new_tokens: int = 64,
) -> float:
    model = trainer.model
    tokenizer = trainer.tokenizer
    model.eval()

    if isinstance(eval_dataset, GSMK8Dataset):
        hf_dataset = convert_to_hf_dataset(eval_dataset)
    else:
        hf_dataset = eval_dataset

    total = 0
    correct = 0

    for start in tqdm(range(0, len(hf_dataset), batch_size), desc="[EVAL]"):
        end = min(start + batch_size, len(hf_dataset))
        batch = hf_dataset.select(range(start, end))

        prompts = batch["prompt"]
        solutions = batch["solution"]

        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for text, sol in zip(texts, solutions):
            if not sol:
                continue
            pred = GSMK8Dataset._extract_final_answer(text)
            if normalize_number(pred) == normalize_number(sol):
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"[EVAL] Accuracy: {acc:.4f} ({correct}/{total})")
    return acc
