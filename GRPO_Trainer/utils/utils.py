import re
import yaml
from typing import List, Any


def normalize_number(ans: str) -> str:
    ans = ans.strip().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", ans)
    if m:
        return m.group(0)
    return ans


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_text_from_completions(completions: List[Any]) -> List[str]:
    texts: List[str] = []
    for comp in completions:
        if isinstance(comp, list) and comp and isinstance(comp[0], dict):
            texts.append(comp[0].get("content", ""))
        else:
            texts.append(str(comp))
    return texts