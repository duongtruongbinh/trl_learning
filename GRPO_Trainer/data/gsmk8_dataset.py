import json
import re
from typing import List, Dict, Optional
from torch.utils.data import Dataset


class GSMK8Dataset(Dataset):
    """
    Input Jsonl format:
    {"question": "...", "answer": "...#### 42"}

    Output:
      {
        "prompt":   <string: question>,
        "solution": <string: final answer, e.g. '42'>,
        "answer":   <string: full chain-of-thought + ####>
      }
    """

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
        """
        Extract the part after '\n#### ' as the final answer.
        Example: '...\\n#### 5' -> '5'
        - Only matches if '####' is immediately after a newline '\n'
        - If there are multiple '\n####', use the last one.
        - If not found, fall back to the last number in the text (if any).
        """
        # 1) Find all segments that appear after '\n####'
        matches = re.findall(r"\n####\s*([^\n]+)", answer_text)
        if matches:
            # If there are multiple matches, take the last one
            return matches[-1].strip()

        # 2) If there is no '\n####', try to take the last number in the whole text
        nums = re.findall(r"-?\d+\.?\d*", answer_text)
        if nums:
            return nums[-1]

        # 3) If nothing is found, return the original string (stripped)
        return answer_text.strip()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]
