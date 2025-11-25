import re
import unittest


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



class TestExtractFinalAnswer(unittest.TestCase):
    def test_case1(self):
        text = "In one hour, there are 3 sets of 20 minutes.\nSo, Joy can read 8 x 3 = <<8*3=24>>24 pages in an hour.\nIt will take her 120\/24 = <<120\/24=5>>5 hours to read 120 pages.\n#### 5"
        self.assertEqual(_extract_final_answer(text), "5")

    def test_case2(self):
        text = "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
        self.assertEqual(_extract_final_answer(text), "70000")

if __name__ == "__main__":
    unittest.main()
