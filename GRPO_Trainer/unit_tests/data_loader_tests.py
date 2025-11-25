import os
import sys
import json
import tempfile
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from data.data_loader import GSMK8Dataset  # <-- change this to your actual module name

class TestGSMK8Dataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary jsonl file for testing
        self.tmp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8")
        self.path = self.tmp_file.name

        # Prepare some sample data
        data = [
            {
                "question": "Q1: simple newline hash",
                "answer": "Reasoning line 1\n#### 5"
            },
            {
                "question": "Q2: multiple hashes but only newline-hash counts",
                "answer": "Result is here #### 3.14   \n#### 42"
            },
            {
                "question": "Q3: no newline-hash, but has numbers",
                "answer": "He tried 3 times, then 7 times, final result is 10"
            },
            {
                "question": "Q4: no hash, no numbers",
                "answer": "No numeric answer here"
            },
        ]

        for obj in data:
            self.tmp_file.write(json.dumps(obj) + "\n")

        self.tmp_file.flush()
        self.tmp_file.close()

    def tearDown(self):
        # Clean up the temporary file after tests
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_length(self):
        dataset = GSMK8Dataset(self.path)
        self.assertEqual(len(dataset), 4)

    def test_item_structure_with_solution(self):
        dataset = GSMK8Dataset(self.path, extract_solution=True)
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("answer", sample)
        self.assertIn("solution", sample)

    def test_item_structure_without_solution(self):
        dataset = GSMK8Dataset(self.path, extract_solution=False)
        sample = dataset[0]
        self.assertIn("prompt", sample)
        self.assertIn("answer", sample)
        self.assertNotIn("solution", sample)

    def test_solution_extraction_newline_hash(self):
        dataset = GSMK8Dataset(self.path, extract_solution=True)

        # Q1: "Reasoning line 1\n#### 5" -> should extract "5"
        self.assertEqual(dataset[0]["solution"], "5")

        # Q2: "Result is here #### 3.14   \n#### 42"
        # Only "\n#### 42" matches, so solution should be "42"
        self.assertEqual(dataset[1]["solution"], "42")

    def test_solution_extraction_fallback_last_number(self):
        dataset = GSMK8Dataset(self.path, extract_solution=True)

        # Q3: No '\n####', but has numbers 3, 7, 10 -> should return "10"
        self.assertEqual(dataset[2]["solution"], "10")

    def test_solution_extraction_no_number_return_full(self):
        dataset = GSMK8Dataset(self.path, extract_solution=True)

        # Q4: No '\n####' and no numbers -> return original answer stripped
        self.assertEqual(dataset[3]["solution"], "No numeric answer here")


if __name__ == "__main__":
    unittest.main()
