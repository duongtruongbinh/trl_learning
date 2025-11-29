import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from data.gsmk8_dataset import GSMK8Dataset
from utils.utils import normalize_number, extract_text_from_completions


def math_accuracy_reward(completions, solution, **kwargs) -> List[float]:
    texts = extract_text_from_completions(completions)
    rewards: List[float] = []
    
    for text, sol in zip(texts, solution):
        pred = GSMK8Dataset._extract_final_answer(text)
        pred_norm = normalize_number(pred)
        sol_norm = normalize_number(sol)
        if pred_norm and sol_norm and pred_norm == sol_norm:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
