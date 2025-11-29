from typing import Dict, Any

from datasets import Dataset
from tqdm.auto import tqdm
from trl import GRPOConfig, GRPOTrainer

from data.gsmk8_dataset import GSMK8Dataset
from trainer.rewards import math_accuracy_reward


def convert_to_hf_dataset(gsmk8_dataset: GSMK8Dataset) -> Dataset:
    data_list = []
    for i in tqdm(range(len(gsmk8_dataset)), desc="Converting GSMK8Dataset to HF Dataset"):
        item = gsmk8_dataset[i]
        data_list.append({
            "prompt": item["prompt"],
            "solution": item.get("solution", ""),
        })
    return Dataset.from_list(data_list)


def train_grpo(cfg: Dict[str, Any], train_dataset: GSMK8Dataset) -> GRPOTrainer:
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    
    model_name = model_cfg.get("name", "Qwen/Qwen2-0.5B-Instruct")
    output_dir = cfg.get("output_dir", "models/checkpoints")

    hf_dataset = convert_to_hf_dataset(train_dataset)

    training_args = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        num_generations=training_cfg.get("num_generations", 2),
        max_prompt_length=training_cfg.get("max_prompt_length", 128),
        max_completion_length=training_cfg.get("max_completion_length", 64),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_steps=training_cfg.get("save_steps", 500),
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        bf16=training_cfg.get("bf16", True),
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=math_accuracy_reward,
        args=training_args,
        train_dataset=hf_dataset,
    )

    trainer.train()

    return trainer
