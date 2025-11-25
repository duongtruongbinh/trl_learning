"""Inference script for Qwen3-VL models (baseline and fine-tuned)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""

    parser = argparse.ArgumentParser(description="Run inference on Qwen3-VL model")
    parser.add_argument("--model_name", required=True, help="Path to model (baseline or fine-tuned)")
    parser.add_argument("--dataset_name", required=True, help="Path to JSON dataset file")
    parser.add_argument("--output_file", required=True, help="Path to save predictions JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (None = all)")

    return parser.parse_args()


def load_model_and_processor(model_path: str):
    """Load Qwen3-VL model and processor."""
    print(f"Loading model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print("Model loaded successfully!")
    return model, processor


def prepare_messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Prepare messages from dataset item for Qwen3-VL format."""
    messages = []
    
    user_content = []
    
    if "images" in item and item["images"]:
        for img_path in item["images"]:
            if isinstance(img_path, str) and Path(img_path).exists():
                user_content.append({"type": "image", "image": img_path})
            elif isinstance(img_path, str):
                print(f"Warning: Image not found: {img_path}")
    
    if "prompt" in item and isinstance(item["prompt"], list):
        for msg in item["prompt"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict):
                            user_content.append(c)
                        elif isinstance(c, str):
                            user_content.append({"type": "text", "text": c})
    
    if user_content:
        messages.append({"role": "user", "content": user_content})
    
    return messages


def generate_answer(model, processor, messages: list[dict], generation_kwargs: dict) -> str:
    """Generate answer from model given messages."""
    try:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        
        return output_text[0] if output_text else ""
    except Exception as e:
        print(f"Error during generation: {e}")
        return ""


def main() -> None:
    """Run inference on dataset."""
    args = parse_args()
    
    model, processor = load_model_and_processor(args.model_name)
    
    print(f"Loading dataset from {args.dataset_name}...")
    dataset = load_dataset("json", data_files=args.dataset_name)["train"]
    
    if args.num_samples:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    print(f"Processing {len(dataset)} samples...")
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
    }
    
    predictions = []
    
    for i, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
        
        messages = prepare_messages(item)
        if not messages:
            print(f"Warning: Skipping item {i} - no valid messages")
            continue
        
        answer = generate_answer(model, processor, messages, generation_kwargs)
        question = ""
        if "prompt" in item and isinstance(item["prompt"], list) and len(item["prompt"]) > 0:
            first_msg = item["prompt"][0]
            if isinstance(first_msg.get("content"), str):
                question = first_msg["content"]
            elif isinstance(first_msg.get("content"), list):
                for c in first_msg["content"]:
                    if isinstance(c, dict) and c.get("type") == "text":
                        question = c.get("text", "")
                        break
        
        ground_truth = ""
        if "chosen" in item and isinstance(item["chosen"], list) and len(item["chosen"]) > 0:
            ground_truth = item["chosen"][0].get("content", "")
        
        prediction = {
            "question_id": item.get("question_id", i),
            "image": item.get("images", [None])[0] if item.get("images") else None,
            "question": question,
            "answer": answer.strip(),
            "ground_truth": ground_truth,
        }
        
        predictions.append(prediction)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()

