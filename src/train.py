"""Entry point for training a DPO model with TRL."""
from __future__ import annotations

import argparse
from pathlib import Path

from trl import DPOConfig, DPOTrainer

from data.preprocessing import PreferenceColumns, fetch_dataset, filter_and_map
from models.qwen import load_policy_and_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DPO training."""

    parser = argparse.ArgumentParser(description="Train a DPO model")
    parser.add_argument("--model_name", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--dataset_name", default="trl-lib/ultrafeedback_binarized")
    parser.add_argument("--output_dir", default="./models/weights/Qwen2-0.5B-DPO")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--loss_type", default="sigmoid")
    parser.add_argument("--use_unsloth", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default="")
    parser.add_argument("--dataset_split", default="train")
    return parser.parse_args()


def main() -> None:
    """Configure the trainer and run training."""

    args = parse_args()
    train_dataset = fetch_dataset(args.dataset_name, args.dataset_split)
    train_dataset = filter_and_map(train_dataset, PreferenceColumns())

    model, tokenizer = load_policy_and_tokenizer(
        args.model_name,
        use_unsloth=args.use_unsloth,
        attn_implementation="flash_attention_2",
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        loss_type=args.loss_type,
        bf16=args.bf16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id or None,
        logging_steps=10,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
