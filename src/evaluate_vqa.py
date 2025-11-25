"""Evaluate VQA predictions using VQA accuracy metric."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def vqa_accuracy(predicted: str, ground_truth: str | list[str]) -> float:
    """Calculate VQA accuracy for a single prediction.
    
    VQA accuracy: min(1, number of humans that gave that answer / 3)
    For single ground truth, returns 1.0 if match, 0.0 otherwise.
    """
    pred_norm = normalize_answer(predicted)
    
    if isinstance(ground_truth, str):
        gt_norm = normalize_answer(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0
    elif isinstance(ground_truth, list):
        if len(ground_truth) == 0:
            return 0.0
        gt_norms = [normalize_answer(gt) for gt in ground_truth if gt]
        if len(gt_norms) == 0:
            return 0.0
        matches = sum(1 for gt_norm in gt_norms if pred_norm == gt_norm)
        return min(1.0, matches / 3.0)
    else:
        return 0.0


def evaluate_predictions(predictions_file: str, ground_truth_file: str | None = None) -> dict[str, Any]:
    """Evaluate predictions against ground truth."""
    
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    
    if ground_truth_file:
        with open(ground_truth_file, "r", encoding="utf-8") as f:
            ground_truths = {item["question_id"]: item for item in json.load(f)}
    else:
        ground_truths = {pred["question_id"]: pred for pred in predictions}
    
    accuracies = []
    detailed_results = []
    
    for pred in predictions:
        qid = pred["question_id"]
        predicted_answer = pred.get("answer", "")
        ground_truth = pred.get("ground_truth", "")
        
        if ground_truth_file and qid in ground_truths:
            ground_truth = ground_truths[qid].get("chosen", [{}])[0].get("content", "")
            if not ground_truth:
                ground_truth = ground_truths[qid].get("answer", "")
        
        if not ground_truth:
            print(f"Warning: No ground truth for question_id {qid}")
            continue
        
        accuracy = vqa_accuracy(predicted_answer, ground_truth)
        accuracies.append(accuracy)
        
        detailed_results.append({
            "question_id": qid,
            "predicted": predicted_answer,
            "ground_truth": ground_truth,
            "accuracy": accuracy,
            "correct": accuracy > 0.0,
        })
    
    overall_accuracy = np.mean(accuracies) if accuracies else 0.0
    
    results = {
        "overall_accuracy": overall_accuracy,
        "total_samples": len(accuracies),
        "correct_predictions": sum(1 for acc in accuracies if acc > 0.0),
        "detailed_results": detailed_results,
    }
    
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    
    parser = argparse.ArgumentParser(description="Evaluate VQA predictions")
    parser.add_argument("--predictions_file", required=True, help="Path to predictions JSON file")
    parser.add_argument("--ground_truth_file", default=None, help="Path to ground truth JSON file (optional)")
    parser.add_argument("--output_file", default=None, help="Path to save evaluation results (optional)")
    
    return parser.parse_args()


def main() -> None:
    """Run evaluation."""
    
    args = parse_args()
    
    print(f"Evaluating predictions from {args.predictions_file}...")
    results = evaluate_predictions(args.predictions_file, args.ground_truth_file)
    
    print("\n" + "="*50)
    print("VQA Evaluation Results")
    print("="*50)
    print(f"Overall Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    print(f"Accuracy: {results['correct_predictions']}/{results['total_samples']} = {results['overall_accuracy']:.4f}")
    print("="*50)
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()

