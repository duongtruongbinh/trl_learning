"""Evaluation script for trained DPO checkpoints."""
from __future__ import annotations

import argparse

from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer

from data.preprocessing import PreferenceColumns, fetch_dataset, filter_and_map
from models.qwen import load_policy_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a DPO model")
    parser.add_argument("--model_name", default="./models/weights/Qwen2-0.5B-DPO")
    parser.add_argument("--dataset_name", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--dataset_split", default="eval")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", default="./models/weights/eval_tmp")
    return parser.parse_args()


def main() -> None:
    """Run evaluation against a preference dataset."""

    args = parse_args()
    dataset = fetch_dataset(args.dataset_name, args.dataset_split)
    dataset = filter_and_map(dataset, PreferenceColumns())

    model, _ = load_policy_and_tokenizer(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    config = DPOConfig(output_dir=args.output_dir, max_length=args.max_length)
    trainer = DPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=None,
        eval_dataset=dataset,
    )

    metrics = trainer.evaluate(max_length=args.max_length)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
