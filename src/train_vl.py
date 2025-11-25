"""Entry point for training Qwen3-VL model with DPO."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, load_dataset
import torch
from peft import LoraConfig, TaskType
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.preprocessing import fetch_dataset


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for VL DPO training."""

    parser = argparse.ArgumentParser(description="Train Qwen3-VL model with DPO")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset_name", default="", help="Dataset name or path to JSON files")
    parser.add_argument("--dataset_split", default="train", help="Dataset split to use")
    parser.add_argument("--output_dir", default="./models/weights/Qwen3-VL-2B-DPO")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--loss_type", default="sigmoid", help="DPO loss type")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_ref_model", action="store_true", help="Use reference model for DPO")
    parser.add_argument("--preprocess_num_proc", type=int, default=None, help="Number of processes for preprocessing (None = all CPUs)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--cache_dir", default=None, help="Cache directory for tokenized dataset")

    return parser.parse_args()


def load_model_and_processor(model_name: str):
    """Load Qwen3-VL model and processor."""
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, processor


def main() -> None:
    """Configure the trainer and run DPO training."""

    args = parse_args()

    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(args.dataset_name).stem if args.dataset_name else "dataset"
        cache_path = cache_dir / f"{dataset_name}_tokenized"
        if cache_path.exists():
            print(f"Loading cached tokenized dataset from {cache_path}")
            train_dataset = Dataset.load_from_disk(str(cache_path))
        else:
            if args.dataset_name:
                if Path(args.dataset_name).exists():
                    train_dataset = load_dataset(
                        "json",
                        data_files=args.dataset_name,
                        num_proc=args.preprocess_num_proc,
                    )[args.dataset_split]
                else:
                    train_dataset = fetch_dataset(args.dataset_name, args.dataset_split)
            else:
                train_dataset = fetch_dataset(split=args.dataset_split)
    else:
        if args.dataset_name:
            if Path(args.dataset_name).exists():
                train_dataset = load_dataset(
                    "json",
                    data_files=args.dataset_name,
                    num_proc=args.preprocess_num_proc,
                )[args.dataset_split]
            else:
                train_dataset = fetch_dataset(args.dataset_name, args.dataset_split)
        else:
            train_dataset = fetch_dataset(split=args.dataset_split)

    peft_config = None
    if args.lora_r > 0:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

    model, processor = load_model_and_processor(args.model_name)
    
    if peft_config:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
    
    ref_model = None
    if args.use_ref_model:
        ref_model, _ = load_model_and_processor(args.model_name)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        beta=args.beta,
        loss_type=args.loss_type,
        bf16=args.bf16,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=args.dataloader_num_workers,
        dataset_num_proc=args.preprocess_num_proc,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )
    
    print(f"Dataset loaded: {len(train_dataset)} samples")
    print(f"Preprocessing with {args.preprocess_num_proc or 'default'} workers")
    print(f"Dataloader using {args.dataloader_num_workers} workers")
    
    trainer.train()
    
    if cache_path and not cache_path.exists():
        print(f"Saving tokenized dataset cache to {cache_path} for faster loading next time")
        try:
            train_dataset.save_to_disk(str(cache_path))
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    trainer.save_model()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
