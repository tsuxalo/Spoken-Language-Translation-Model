import unittest
from pathlib import Path

import nbformat

from scripts.validate_notebook import validate_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/00_data_loading_preprocessing.ipynb"
DIRECT_NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/02_direct_s2tt_training.ipynb"


class NotebookContractTests(unittest.TestCase):
    def test_notebook_is_small_clean_valid_and_compilable(self):
        result = validate_notebook(NOTEBOOK_PATH, max_size_bytes=250_000)
        self.assertTrue(result["outputs_empty"])
        self.assertTrue(result["execution_counts_empty"])
        self.assertEqual(result["secret_scan"], "passed")

    def test_required_narrative_and_safe_defaults_are_present(self):
        notebook = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
        markdown = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        code = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        for section in range(1, 23):
            self.assertIn(f"{section}.", markdown)
        self.assertIn('REPO_REF = "feature/direct-s2tt"', code)
        self.assertIn("RUN_FULL_METADATA_AUDIT = False", code)
        self.assertIn("BUILD_FULL_TRAINING_DATASET = False", code)
        self.assertIn("WRITE_ARTIFACTS = False", code)
        self.assertNotIn('load_naija_split("dev"', code)
        self.assertNotIn("git reset --hard", code)

    def test_direct_training_notebook_contract_and_stage_gates(self):
        result = validate_notebook(DIRECT_NOTEBOOK_PATH, max_size_bytes=250_000)
        self.assertTrue(result["outputs_empty"])
        self.assertTrue(result["execution_counts_empty"])
        self.assertEqual(result["secret_scan"], "passed")
        notebook = nbformat.read(str(DIRECT_NOTEBOOK_PATH), as_version=4)
        markdown_text = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        code_text = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        positions = [markdown_text.index(f"## {section}.") for section in range(1, 34)]
        self.assertEqual(positions, sorted(positions))
        for flag in (
            "RUN_STRUCTURAL_SMOKE = False",
            "RUN_BASELINE_EVALUATION = False",
            "RUN_MATCHED_PILOTS = False",
            "RUN_FULL_TRAINING = False",
        ):
            self.assertIn(flag, code_text)
        self.assertIn('REPO_REF = "feature/direct-s2tt-training-notebook"', code_text)
        self.assertIn("official_dev_evaluated", markdown_text + code_text)
        self.assertNotIn('load_naija_split("dev"', code_text)
        self.assertNotIn("git reset --hard", code_text)


if __name__ == "__main__":
    unittest.main()
