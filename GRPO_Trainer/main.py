import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from trainer.eval import evaluate_accuracy
from trainer.train import train_grpo
from utils.utils import load_config
from data.gsmk8_dataset import GSMK8Dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/grpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_cfg = cfg.get("dataset", {})
    train_path = dataset_cfg.get("train_path", "data/raw/train.jsonl")
    val_path = dataset_cfg.get("val_path", "data/raw/test.jsonl")

    print(f"[CONFIG] Using train_path = {train_path}")
    print(f"[CONFIG] Using val_path   = {val_path}")

    train_dataset = GSMK8Dataset(train_path)
    val_dataset = GSMK8Dataset(val_path)

    print("[TRAIN] Start GRPO training...")
    trainer = train_grpo(cfg, train_dataset=train_dataset)

    print("[EVAL] Evaluating on validation set...")
    val_acc = evaluate_accuracy(trainer, val_dataset)
    print(f"[EVAL] Validation accuracy = {val_acc:.4f}")


if __name__ == "__main__":
    main()
