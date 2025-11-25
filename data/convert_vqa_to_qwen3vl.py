"""Convert VQA datasets to Qwen3-VL training format."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def resolve_image_path(image_path: str, dataset_name: str) -> str | None:
    """Resolve relative image path to absolute path."""
    base_paths = {
        "vqav2": {
            "train": "/mnt/VLAI_data/COCO_Images/train2014",
            "val": "/mnt/VLAI_data/VQAv2/val2014",
        },
        "textvqa": {
            "train": "/mnt/VLAI_data/TextVQA/train_images",
            "val": "/mnt/VLAI_data/TextVQA/train_images",  # TextVQA uses same folder
        },
        "okvqa": {
            "train": "/mnt/VLAI_data/COCO_Images/train2014",
            "val": "/mnt/VLAI_data/VQAv2/val2014",
        },
    }

    if dataset_name.lower() not in base_paths:
        return None

    image_name = Path(image_path).name
    split = "train" if "train" in image_path.lower() else "val"
    base_path = base_paths[dataset_name.lower()][split]
    full_path = Path(base_path) / image_name

    if full_path.exists():
        return str(full_path)
    return None


def create_rejected_answer(chosen_answer: str | list) -> str:
    """Create a rejected answer from chosen answer for preference dataset."""
    if isinstance(chosen_answer, list):
        chosen_answer = chosen_answer[0] if chosen_answer else "yes"
    if not isinstance(chosen_answer, str):
        chosen_answer = str(chosen_answer)
    
    chosen_lower = chosen_answer.lower().strip()
    
    if len(chosen_answer) < 10:
        rejected = "yes" if chosen_lower not in ["yes", "no"] else "no"
    elif any(word in chosen_lower for word in ["yes", "yeah", "yep"]):
        rejected = "no"
    elif any(word in chosen_lower for word in ["no", "nope", "not"]):
        rejected = "yes"
    else:
        words = chosen_answer.split()
        if len(words) > 3:
            rejected = " ".join(words[:2]) + "..."
        else:
            rejected = "I don't know."
    
    return rejected


def convert_vqa_to_dpo_preference_format(
    jsonl_path: str,
    dataset_name: str,
    output_path: str,
    max_samples: int | None = None,
) -> None:
    """Convert VQA JSONL to DPO preference format for Qwen3-VL."""
    output_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if max_samples:
            lines = lines[:max_samples]

        for line in tqdm(lines, desc=f"Converting {dataset_name}"):
            item = json.loads(line.strip())

            image_path = item.get("image", "")
            question = item.get("question", "")
            answer = item.get("answer", "")

            if not image_path or not question or not answer:
                continue

            resolved_path = resolve_image_path(image_path, dataset_name)
            if not resolved_path:
                continue
            
            user_content = f"Answer the following question based solely on the image content with a single term.\nQuestion: {question}"
            dpo_format = {
                "images": [resolved_path],
                "prompt": [
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                ],
                "rejected": [
                    {
                        "role": "assistant",
                        "content": create_rejected_answer(answer),
                    }
                ],
            }

            output_data.append(dpo_format)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(output_data)} samples to {output_path}")


def main():
    """Convert multiple VQA datasets."""
    datasets = [
        {
            "name": "vqav2",
            "train": "/mnt/VLAI_data/VQAv2/vqav2_train.jsonl",
            "val": "/mnt/VLAI_data/VQAv2/vqav2_val.jsonl",
        },
        {
            "name": "textvqa",
            "train": "/mnt/VLAI_data/TextVQA/textvqa_train.jsonl",
            "val": "/mnt/VLAI_data/TextVQA/textvqa_val.jsonl",
        },
        {
            "name": "okvqa",
            "train": "/mnt/VLAI_data/OKVQA/okvqa_train.jsonl",
            "val": "/mnt/VLAI_data/OKVQA/okvqa_val.jsonl",
        },
    ]

    output_dir = Path(__file__).parent.parent / "data" / "data_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for split in ["train", "val"]:
            jsonl_path = dataset[split]
            if not Path(jsonl_path).exists():
                print(f"Skipping {jsonl_path} (not found)")
                continue

            output_path = output_dir / f"{dataset['name']}_{split}_dpo.json"
            convert_vqa_to_dpo_preference_format(
                jsonl_path,
                dataset["name"],
                str(output_path),
                max_samples=None,  # Set to number if you want to limit
            )


if __name__ == "__main__":
    main()



