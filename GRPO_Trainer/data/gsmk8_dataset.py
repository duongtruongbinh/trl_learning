import json
import re
from typing import List, Dict
from torch.utils.data import Dataset


class GSMK8Dataset(Dataset):
    def __init__(
        self,
        path: str,
        question_key: str = "question",
        answer_key: str = "answer",
        extract_solution: bool = True,
    ):
        self.path = path
        self.question_key = question_key
        self.answer_key = answer_key
        self.extract_solution = extract_solution

        self.samples: List[Dict] = []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                question = obj[self.question_key]
                answer = obj[self.answer_key]

                if self.extract_solution:
                    solution = self._extract_final_answer(answer)
                else:
                    solution = None

                sample = {
                    "prompt": question,
                    "answer": answer,
                }
                if solution is not None:
                    sample["solution"] = solution

                self.samples.append(sample)

    @staticmethod
    def _extract_final_answer(answer_text: str) -> str:
        matches = re.findall(r"\n####\s*([^\n]+)", answer_text)
        if matches:
            return matches[-1].strip()

        nums = re.findall(r"-?\d+\.?\d*", answer_text)
        if nums:
            return nums[-1]

        return answer_text.strip()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
    
class GSMK8Collator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        prompts   = [ex["prompt"] for ex in batch]
        answers   = [ex.get("answer", "") for ex in batch]
        solutions = [ex.get("solution", "") for ex in batch]

        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompt": prompts,
            "answer": answers,
            "solution": solutions,
        }

