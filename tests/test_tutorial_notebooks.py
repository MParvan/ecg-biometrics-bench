import io
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import load_dataset
import main
import models
import run
import utils

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

FRAMEWORK_MODULES = {
    "load_dataset": load_dataset,
    "run": run,
    "models": models,
    "utils": utils,
    "main": main,
}

# Symbols a tutorial legitimately defines itself rather than importing.
TUTORIAL_PLACEHOLDERS = {"MyAwesomeECGNet"}

# API names removed in earlier revisions. A tutorial mentioning one of these
# is stale and would fail for a reader following it.
REMOVED_API_NAMES = (
    "run_verification",
    "load_all_sessions",
    "enrollment_mode",
    "preprocessing_params=",
    "num_beats=",
    "load_session(\"Session_",
)


def notebook_paths():
    return sorted(EXPERIMENTS_DIR.glob("*.ipynb"))


def notebook_source(path):
    """
    Return the concatenated source of every cell in a notebook.
    """
    notebook = json.load(io.open(path, encoding="utf-8"))

    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )


def notebook_code(path):
    """
    Return the concatenated source of the code cells only.
    """
    notebook = json.load(io.open(path, encoding="utf-8"))

    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


class NotebookIntegrityTests(unittest.TestCase):
    """
    Every shipped notebook must be valid and non-empty.
    """

    def test_notebooks_are_present(self):
        self.assertTrue(notebook_paths())

    def test_every_notebook_is_valid_json(self):
        for path in notebook_paths():
            with self.subTest(notebook=path.name):
                notebook = json.load(
                    io.open(path, encoding="utf-8")
                )

                self.assertIn("cells", notebook)
                self.assertTrue(notebook["cells"])


class StaleApiTests(unittest.TestCase):
    """
    A tutorial that references a removed API would fail for its reader.
    """

    def test_no_notebook_uses_a_removed_api(self):
        for path in notebook_paths():
            source = notebook_source(path)

            for removed_name in REMOVED_API_NAMES:
                with self.subTest(
                    notebook=path.name,
                    api=removed_name,
                ):
                    self.assertNotIn(
                        removed_name,
                        source,
                        f"{path.name} references the removed API "
                        f"'{removed_name}'.",
                    )

    def test_no_broken_dataset_scripts_remain(self):
        stale_scripts = list(
            EXPERIMENTS_DIR.glob("*_gather_results.py")
        )

        self.assertEqual(
            stale_scripts,
            [],
            "The dataset gather-results scripts were removed because they "
            "called a non-existent API and hard-coded 2 training epochs.",
        )


class ImportResolutionTests(unittest.TestCase):
    """
    Every framework symbol a notebook imports must still exist.
    """

    def test_all_framework_imports_resolve(self):
        for path in notebook_paths():
            code = notebook_code(path)
            unresolved = []

            for line in code.splitlines():
                line = line.strip()

                if not line.startswith("from ") or " import " not in line:
                    continue

                module_name = line.split()[1]

                if module_name not in FRAMEWORK_MODULES:
                    continue

                imported = line.split(" import ", 1)[1]
                imported = imported.strip().strip("()")

                for symbol in imported.replace("\\", "").split(","):
                    symbol = symbol.strip()

                    if not symbol:
                        continue

                    # "X as Y" imports X.
                    symbol = symbol.split(" as ")[0].strip()

                    if symbol in TUTORIAL_PLACEHOLDERS:
                        continue

                    if not hasattr(
                        FRAMEWORK_MODULES[module_name],
                        symbol,
                    ):
                        unresolved.append(
                            f"{module_name}.{symbol}"
                        )

            with self.subTest(notebook=path.name):
                self.assertEqual(
                    unresolved,
                    [],
                    f"{path.name} imports symbols that no longer exist: "
                    f"{unresolved}",
                )


class CustomModelTutorialTests(unittest.TestCase):
    """
    The registration tutorial must describe the mechanism that exists.
    """

    def setUp(self):
        self.source = notebook_source(
            EXPERIMENTS_DIR / "Custom_Model.ipynb"
        )

    def test_tutorial_uses_the_real_registry(self):
        self.assertIn("MODEL_REGISTRY", self.source)

    def test_tutorial_does_not_describe_a_nonexistent_helper(self):
        self.assertNotIn("get_model", self.source)

    def test_registry_named_by_the_tutorial_exists(self):
        self.assertTrue(
            hasattr(main, "MODEL_REGISTRY")
        )
        self.assertIsInstance(
            main.MODEL_REGISTRY,
            dict,
        )

    def test_tutorial_states_the_length_independence_requirement(self):
        # This is the requirement a custom model is most likely to violate.
        self.assertIn(
            "AdaptiveAvgPool1d",
            self.source,
        )

    def test_tutorial_states_the_include_top_contract(self):
        self.assertIn("include_top", self.source)


class TutorialCoverageTests(unittest.TestCase):
    """
    The tutorials must cover the three core workflows.
    """

    def test_dataset_loading_tutorial_exists(self):
        self.assertTrue(
            (
                EXPERIMENTS_DIR / "load_dataset_Module.ipynb"
            ).exists()
        )

    def test_protocol_switching_tutorial_exists(self):
        self.assertTrue(
            (EXPERIMENTS_DIR / "run_Module.ipynb").exists()
        )

    def test_custom_model_tutorial_exists(self):
        self.assertTrue(
            (EXPERIMENTS_DIR / "Custom_Model.ipynb").exists()
        )

    def test_folder_readme_exists(self):
        self.assertTrue(
            (EXPERIMENTS_DIR / "README.md").exists()
        )


if __name__ == "__main__":
    unittest.main()
