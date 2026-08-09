import unittest
from pathlib import Path

import nbformat

from scripts.validate_notebook import validate_notebook

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks/00_data_loading_preprocessing.ipynb"


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
        code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        for section in range(1, 23):
            self.assertIn(f"{section}.", markdown)
        self.assertIn('REPO_REF = "feature/data-preprocessing-notebook"', code)
        self.assertIn("RUN_FULL_METADATA_AUDIT = False", code)
        self.assertIn("BUILD_FULL_TRAINING_DATASET = False", code)
        self.assertIn("WRITE_ARTIFACTS = False", code)
        self.assertNotIn('load_naija_split("dev"', code)
        self.assertNotIn("git reset --hard", code)


if __name__ == "__main__":
    unittest.main()
