import ast
import io
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CORE_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
OPTIONAL_REQUIREMENTS = (
    PROJECT_ROOT / "requirements-representations.txt"
)

# Distribution name to importable module name, where the two differ.
DISTRIBUTION_TO_MODULE = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "patool": "patoolib",
    "pywavelets": "pywt",
}

# Modules imported lazily inside functions, so they are optional at runtime.
OPTIONAL_MODULES = {
    "librosa",
    "pyts",
    "stockwell",
    "tftb",
}

# Third-party modules used only by the test suite or tooling.
TEST_ONLY_MODULES = {"pytest"}


def read_requirements(path):
    """
    Return the distribution names declared in a requirements file.
    """
    names = []

    for line in io.open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()

        if not line:
            continue

        for separator in ("==", ">=", "<=", "~=", ">", "<"):
            if separator in line:
                line = line.split(separator)[0]
                break

        names.append(line.strip())

    return names


def module_level_imports(path):
    """
    Return third-party modules imported at module level in one file.

    Imports nested inside a function or class body are excluded, because a
    missing package only fails when that code path runs.
    """
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    modules = set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])

    return modules


def framework_source_files():
    """
    Return the framework modules a user imports directly.
    """
    return sorted(PROJECT_ROOT.glob("*.py")) + sorted(
        (PROJECT_ROOT / "scripts").glob("*.py")
    )


LOCAL_MODULES = {
    "main",
    "run",
    "utils",
    "models",
    "load_dataset",
    "preprocessing",
    "filtering",
    "representation",
    "data_augmentation",
    "visualizations",
    "scripts",
}


class CoreRequirementsTests(unittest.TestCase):
    """
    A clean install of requirements.txt must be able to run the benchmark.
    """

    def setUp(self):
        self.declared = read_requirements(CORE_REQUIREMENTS)
        self.declared_modules = {
            DISTRIBUTION_TO_MODULE.get(
                name.lower(),
                name.lower(),
            )
            for name in self.declared
        }

    def test_every_module_level_import_is_declared(self):
        # A module-level import of an undeclared package makes the framework
        # unusable immediately after a clean install.
        undeclared = {}

        for path in framework_source_files():
            for module in module_level_imports(path):
                if module in sys.stdlib_module_names:
                    continue
                if module in LOCAL_MODULES:
                    continue
                if module in TEST_ONLY_MODULES:
                    continue
                if module.lower() in self.declared_modules:
                    continue

                undeclared.setdefault(module, []).append(
                    path.name
                )

        self.assertEqual(
            undeclared,
            {},
            "These modules are imported at module level but are not in "
            f"requirements.txt: {undeclared}",
        )

    def test_every_declared_requirement_is_importable(self):
        import importlib

        for name in self.declared:
            module = DISTRIBUTION_TO_MODULE.get(
                name.lower(),
                name.lower(),
            )

            with self.subTest(requirement=name):
                try:
                    importlib.import_module(module)
                except ImportError as error:
                    self.fail(
                        f"'{name}' is declared in requirements.txt but "
                        f"cannot be imported: {error}"
                    )

    def test_every_requirement_is_version_pinned(self):
        unpinned = []

        for line in io.open(
            CORE_REQUIREMENTS,
            encoding="utf-8",
        ):
            line = line.split("#")[0].strip()

            if line and "==" not in line:
                unpinned.append(line)

        self.assertEqual(
            unpinned,
            [],
            f"Unpinned requirements: {unpinned}",
        )

    def test_optional_packages_are_not_core_requirements(self):
        # A failed build of an optional package must not prevent the
        # framework from installing.
        for module in OPTIONAL_MODULES:
            with self.subTest(module=module):
                self.assertNotIn(
                    module,
                    self.declared_modules,
                )


class OptionalRequirementsTests(unittest.TestCase):
    """
    Optional packages are declared separately and imported lazily.
    """

    def test_optional_requirements_file_exists(self):
        self.assertTrue(OPTIONAL_REQUIREMENTS.exists())

    def test_optional_file_declares_every_optional_package(self):
        declared = {
            name.lower()
            for name in read_requirements(
                OPTIONAL_REQUIREMENTS
            )
        }

        for module in OPTIONAL_MODULES:
            with self.subTest(module=module):
                self.assertIn(module, declared)

    def test_optional_packages_are_never_imported_at_module_level(self):
        # If one of these moved to a module-level import, the core install
        # would silently gain a hard dependency on it.
        offenders = {}

        for path in framework_source_files():
            for module in module_level_imports(path):
                if module in OPTIONAL_MODULES:
                    offenders.setdefault(
                        module,
                        [],
                    ).append(path.name)

        self.assertEqual(
            offenders,
            {},
            "Optional packages imported at module level: "
            f"{offenders}",
        )

    def test_optional_requirements_are_pinned(self):
        unpinned = []

        for line in io.open(
            OPTIONAL_REQUIREMENTS,
            encoding="utf-8",
        ):
            line = line.split("#")[0].strip()

            if line and "==" not in line:
                unpinned.append(line)

        self.assertEqual(unpinned, [])


class DeclaredRequirementUsageTests(unittest.TestCase):
    """
    Declared packages should actually be used somewhere in the framework.
    """

    def test_no_unused_core_requirement(self):
        used = set()

        for path in framework_source_files():
            tree = ast.parse(
                io.open(path, encoding="utf-8").read()
            )

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        used.add(
                            alias.name.split(".")[0].lower()
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        used.add(
                            node.module.split(".")[0].lower()
                        )

        unused = []

        for name in read_requirements(CORE_REQUIREMENTS):
            module = DISTRIBUTION_TO_MODULE.get(
                name.lower(),
                name.lower(),
            )

            if module not in used:
                unused.append(name)

        self.assertEqual(
            unused,
            [],
            f"Declared but never imported: {unused}",
        )


if __name__ == "__main__":
    unittest.main()
