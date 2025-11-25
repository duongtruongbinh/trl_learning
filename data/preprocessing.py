"""Utilities to load, clean, and format preference data for DPO."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_DATASET = "trl-lib/ultrafeedback_binarized"
RAW_DIR = Path(__file__).resolve().parent / "data_raw"


@dataclass
class PreferenceColumns:
    """Column mapping for prompt, chosen, and rejected texts.
    
    For multimodal VLM datasets, also supports 'images' field.
    """

    prompt: str = "prompt"
    chosen: str = "chosen"
    rejected: str = "rejected"
    images: str = "images"


def fetch_dataset(dataset_name: str = DEFAULT_DATASET, split: str = "train") -> Dataset:
    """Download a dataset via HF hub or load from disk."""

    if RAW_DIR.exists():
        disk_files = list(RAW_DIR.glob("*.json"))
        if disk_files:
            split_files = [
                str(p)
                for p in disk_files
                if (split in p.name or (split == "train" and "train" in p.name))
                and "_dpo" in p.name
            ]
            if split_files:
                return load_dataset("json", data_files={split: split_files})[split]
    return load_dataset(dataset_name, split=split)


def filter_and_map(
    dataset: Dataset,
    columns: PreferenceColumns = PreferenceColumns(),
    predicate: Callable[[dict[str, str]], bool] | None = None,
) -> Dataset:
    """Apply optional filtering and rename fields to DPOTrainer defaults."""

    if predicate:
        dataset = dataset.filter(predicate)
    return dataset.rename_columns(
        {
            columns.prompt: "prompt",
            columns.chosen: "chosen",
            columns.rejected: "rejected",
        }
    )


def build_dataset_dict(
    train: Dataset,
    eval_dataset: Dataset | None = None,
) -> DatasetDict:
    """Return a DatasetDict ready for the trainer."""

    data = {"train": train}
    if eval_dataset:
        data["eval"] = eval_dataset
    return DatasetDict(data)


def yield_batched_records(dataset: Dataset, batch_size: int = 8) -> Iterable[list[dict[str, str]]]:
    """Yield dataset batches for custom preprocessing pipelines."""

    batch: list[dict[str, str]] = []
    for item in dataset:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
