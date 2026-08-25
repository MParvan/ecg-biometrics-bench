import codecs
import importlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import artifact_provenance as provenance
from utils import _generate_config_hash


class CanonicalSourceIdentityTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self):
        first = {"b": [2, 3], "a": 1}
        second = {"a": 1, "b": [2, 3]}
        self.assertEqual(
            provenance.canonical_json_bytes(first),
            provenance.canonical_json_bytes(second),
        )
        self.assertEqual(
            provenance.canonical_json_sha256(first),
            provenance.canonical_json_sha256(second),
        )

    def test_line_endings_and_utf8_bom_are_canonical(self):
        lf = b"value = 1\nprint(value)\n"
        variants = (
            lf,
            lf.replace(b"\n", b"\r\n"),
            lf.replace(b"\n", b"\r"),
            codecs.BOM_UTF8 + lf,
        )

        digests = {
            provenance.python_source_identity(
                "example",
                source,
            )["sha256"]
            for source in variants
        }
        self.assertEqual(len(digests), 1)

    def test_real_source_edit_changes_digest(self):
        original = provenance.python_source_identity(
            "example",
            b"value = 1\n",
        )
        edited = provenance.python_source_identity(
            "example",
            b"value = 2\n",
        )
        self.assertNotEqual(original["sha256"], edited["sha256"])

    def test_absolute_location_does_not_enter_source_digest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first" / "component.py"
            second = root / "second" / "component.py"
            first.parent.mkdir()
            second.parent.mkdir()
            source = b"def transform(value):\n    return value + 1\n"
            first.write_bytes(source)
            second.write_bytes(source)

            first_identity = provenance.python_source_identity(
                "component",
                first.read_bytes(),
            )
            second_identity = provenance.python_source_identity(
                "component",
                second.read_bytes(),
            )
            self.assertEqual(first_identity, second_identity)

    def test_logical_name_participates_in_digest(self):
        source = b"value = 1\n"
        first = provenance.python_source_identity("first", source)
        second = provenance.python_source_identity("second", source)
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_zipimport_loader_source_can_be_read_and_hashed(self):
        module_name = "zip_identity_fixture"
        source = b"def result():\r\n    return 7\r\n"

        with tempfile.TemporaryDirectory() as temporary_directory:
            archive_path = Path(temporary_directory) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{module_name}.py", source)

            sys.path.insert(0, str(archive_path))
            importlib.invalidate_caches()
            try:
                loaded = provenance.read_module_source_bytes(module_name)
                identity = provenance.python_source_identity(
                    module_name,
                    loaded,
                )
            finally:
                sys.path.remove(str(archive_path))
                sys.modules.pop(module_name, None)
                importlib.invalidate_caches()

        self.assertEqual(loaded, source)
        self.assertEqual(len(identity["sha256"]), 64)

    def test_missing_required_source_fails_clearly(self):
        with self.assertRaisesRegex(
            provenance.ImplementationSourceError,
            "Required implementation source",
        ):
            provenance.build_implementation_group(
                "missing",
                (("missing", "module_that_cannot_exist_9f4a"),),
            )


class GroupedIdentityTests(unittest.TestCase):
    @staticmethod
    def _source_reader(module_name):
        return f"MODULE = {module_name!r}\n".encode("utf-8")

    def test_implementation_groups_are_exact_and_exclude_nonruntime_text(self):
        self.assertEqual(
            tuple(name for name, _ in provenance.DATA_IMPLEMENTATION_MODULES),
            ("load_dataset", "preprocessing", "filtering"),
        )
        self.assertEqual(
            tuple(name for name, _ in provenance.WEIGHT_IMPLEMENTATION_MODULES),
            ("models", "run", "utils", "data_augmentation"),
        )

        all_names = {
            name
            for name, _ in (
                provenance.DATA_IMPLEMENTATION_MODULES
                + provenance.WEIGHT_IMPLEMENTATION_MODULES
            )
        }
        self.assertNotIn("main", all_names)
        self.assertNotIn("README", all_names)
        self.assertNotIn("plotting", all_names)
        self.assertNotIn("visualizations", all_names)

    def test_weight_identity_aggregates_complete_data_identity(self):
        identity = provenance.build_weight_implementation_identity(
            source_reader=self._source_reader,
        )
        self.assertEqual(
            set(identity["data_implementation"]["components"]),
            {"load_dataset", "preprocessing", "filtering"},
        )
        self.assertEqual(
            set(identity["weight_training_implementation"]["components"]),
            {"models", "run", "utils", "data_augmentation"},
        )

    def test_dirty_source_change_changes_compatibility_uid(self):
        sources = {
            name: self._source_reader(module_name)
            for name, module_name in provenance.DATA_IMPLEMENTATION_MODULES
        }

        def reader(module_name):
            return sources[module_name]

        first = provenance.build_data_implementation_identity(reader)
        sources["filtering"] = b"MODULE = 'filtering'\nEDITED = True\n"
        second = provenance.build_data_implementation_identity(reader)

        first_uid = _generate_config_hash(
            {"implementation_identity": first}
        )
        second_uid = _generate_config_hash(
            {"implementation_identity": second}
        )
        self.assertNotEqual(first_uid, second_uid)


class DependencyIdentityTests(unittest.TestCase):
    def test_data_dependency_mapping_is_complete(self):
        identity = provenance.build_data_dependency_identity(
            version_resolver=lambda name: f"version-for-{name}",
        )
        self.assertEqual(
            tuple(identity["distributions"]),
            ("numpy", "scipy", "neurokit2", "wfdb", "pandas"),
        )

    def test_weight_dependencies_include_data_digest_and_torch(self):
        identity = provenance.build_weight_dependency_identity(
            version_resolver=lambda name: f"version-for-{name}",
        )
        self.assertEqual(
            set(identity["data_dependencies"]["distributions"]),
            {"numpy", "scipy", "neurokit2", "wfdb", "pandas"},
        )
        self.assertEqual(
            identity["weight_training_dependencies"]["distributions"],
            {
                "torch": "version-for-torch",
                "torchvision": "version-for-torchvision",
                "scikit-learn": "version-for-scikit-learn",
            },
        )

    def test_dependency_change_changes_compatibility_uid(self):
        first = provenance.build_data_dependency_identity(
            version_resolver=lambda name: "1.0",
        )
        second = provenance.build_data_dependency_identity(
            version_resolver=lambda name: (
                "2.0" if name == "scipy" else "1.0"
            ),
        )
        self.assertNotEqual(
            _generate_config_hash({"dependency_identity": first}),
            _generate_config_hash({"dependency_identity": second}),
        )


class CreationProvenanceTests(unittest.TestCase):
    def test_available_git_fields_are_collected(self):
        outputs = {
            ("rev-parse", "HEAD"): "abc123",
            ("rev-parse", "--abbrev-ref", "HEAD"): "feature",
            ("status", "--porcelain"): " M run.py",
        }
        creation = provenance.collect_creation_provenance(
            git_runner=lambda root, arguments: outputs[tuple(arguments)],
        )
        self.assertEqual(
            creation,
            {
                "git": {
                    "status": "available",
                    "commit": "abc123",
                    "branch": "feature",
                    "dirty": True,
                }
            },
        )

    def test_git_unavailable_is_explicit_and_content_identity_still_works(self):
        creation = provenance.collect_creation_provenance(
            git_runner=lambda root, arguments: None,
        )
        implementation = provenance.build_data_implementation_identity(
            source_reader=lambda module: b"VALUE = 1\n",
        )

        self.assertEqual(creation["git"]["status"], "unavailable")
        self.assertEqual(creation["git"]["dirty"], "unavailable")
        self.assertEqual(len(implementation["aggregate_sha256"]), 64)

    def test_git_provenance_change_alone_does_not_change_uid(self):
        cache_identity = {
            "dataset": "synthetic",
            "implementation_identity": {"aggregate_sha256": "abc"},
        }
        first_creation = {
            "git": {"commit": "one", "branch": "main", "dirty": False}
        }
        second_creation = {
            "git": {"commit": "two", "branch": "other", "dirty": True}
        }

        first_uid = _generate_config_hash(cache_identity)
        second_uid = _generate_config_hash(cache_identity)
        self.assertNotEqual(first_creation, second_creation)
        self.assertEqual(first_uid, second_uid)


if __name__ == "__main__":
    unittest.main()
