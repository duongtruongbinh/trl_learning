# utils/format_utils.py
import re

def normalize_number(ans: str) -> str:
    """
    Normalize numeric answers so that '42', '42.0', '  42 ' become comparable.
    Very simple heuristic: keep only the first integer/float pattern.
    """
    ans = ans.strip().replace(",", "")
    m = re.search(r"-?\d+(\.\d+)?", ans)
    if m:
        return m.group(0)
    return ans
