"""
Pin how CYBHi selects recordings from its two simultaneous acquiring units.

Every short-term acquisition was recorded at once by unit 8B, from the hand
palms with Ag/AgCl electrodes, and unit 85, from the fingers with electrolycra.
The electrolycra recordings mostly defeat beat detection, so the loader reads
one unit at a time, Ag/AgCl by default. The long-term collection has a single
unit and is unaffected.

The collection a recording belongs to is decided by the directory names on its
path, never by substrings of the whole path, which would make the result
depend on where the repository is cloned.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import utils
from load_dataset import load_cybhi_dataset


def write_recording(root, collection, filename, seed, columns):
    """
    Write one recording in the format its collection uses: seven columns for
    short-term bioPlux files, one for long-term.
    """
    directory = root / "CYBHi" / "data" / collection
    directory.mkdir(parents=True, exist_ok=True)

    generator = np.random.default_rng(seed)
    samples = generator.integers(0, 4096, size=(2000, columns))

    (directory / filename).write_text(
        "\n".join(" ".join(str(v) for v in row) for row in samples) + "\n"
    )


def build_dataset(root):
    # Short-term: each acquisition present on both units.
    write_recording(root, "short-term", "20110715-MLS-CI-8B.txt", 1, 7)
    write_recording(root, "short-term", "20110715-MLS-CI-85.txt", 2, 7)
    write_recording(root, "short-term", "20110715-MLS-A1-8B.txt", 3, 7)
    write_recording(root, "short-term", "20110715-MLS-A1-85.txt", 4, 7)

    # A test acquisition that is not a participant.
    write_recording(root, "short-term", "20110718-VIDEOPRINT-A1-8B.txt", 5, 7)

    # Long-term: two visits of one subject, single unit.
    write_recording(root, "long-term", "20120106-AA-A0-35.txt", 6, 1)
    write_recording(root, "long-term", "20120416-AA-A0-35.txt", 7, 1)


def loaded_tags(loader):
    recordings = loader.load_raw_data()
    return {
        sid: {tag: len(recs) for tag, recs in sessions.items() if recs}
        for sid, sessions in recordings.items()
    }


class CYBHiUnitSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        build_dataset(self.root)

    def loader(self, **kwargs):
        loader = load_cybhi_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["short-term_CI"],
            **kwargs,
        )
        loader.dataset_root = self.root
        return loader

    def test_the_default_reads_the_silver_chloride_unit(self):
        loader = self.loader()

        self.assertEqual(loader.electrode_unit, "8B")

        tags = loaded_tags(loader)
        self.assertEqual(tags["MLS"]["short-term_CI"], 1)
        self.assertEqual(tags["MLS"]["short-term_A1"], 1)

    def test_the_electrolycra_unit_is_selectable(self):
        tags = loaded_tags(self.loader(electrode_unit="85"))

        self.assertEqual(tags["MLS"]["short-term_CI"], 1)

    def test_both_units_can_be_pooled_for_an_electrode_comparison(self):
        tags = loaded_tags(self.loader(electrode_unit="both"))

        self.assertEqual(tags["MLS"]["short-term_CI"], 2)
        self.assertEqual(tags["MLS"]["short-term_A1"], 2)

    def test_the_long_term_collection_is_unaffected_by_the_unit(self):
        for unit in ("8B", "85", "both"):
            with self.subTest(unit=unit):
                tags = loaded_tags(self.loader(electrode_unit=unit))

                self.assertEqual(tags["AA"]["long-term_S1"], 1)
                self.assertEqual(tags["AA"]["long-term_S2"], 1)

    def test_an_unknown_unit_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "electrode_unit must be one of",
        ):
            self.loader(electrode_unit="35")

    def test_the_test_acquisition_is_not_a_subject(self):
        tags = loaded_tags(self.loader())

        self.assertNotIn("VIDEOPRINT", tags)

    def test_the_unit_reaches_the_cache_identity(self):
        """
        The two units record different electrodes, so arrays built from one
        must not be served to a run that asked for the other.
        """
        hashes = {
            utils._generate_config_hash(
                utils._build_loader_cache_identity(self.loader(electrode_unit=unit))
            )
            for unit in ("8B", "85", "both")
        }

        self.assertEqual(len(hashes), 3)


class CYBHiPathIndependenceTests(unittest.TestCase):
    def test_a_clone_path_containing_ci_does_not_change_the_pools(self):
        """
        The collection must follow from the directory names inside the
        dataset, not from the path the repository sits in.
        """
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base, ignore_errors=True)

        root = base / "cibm-revision" / "datasets" / "cybhi"
        build_dataset(root)

        loader = load_cybhi_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["long-term_S1"],
        )
        loader.dataset_root = root

        tags = loaded_tags(loader)

        self.assertIn("long-term_S1", tags["AA"])
        self.assertIn("long-term_S2", tags["AA"])
        self.assertNotIn("short-term_A0", tags.get("AA", {}))


if __name__ == "__main__":
    unittest.main()
