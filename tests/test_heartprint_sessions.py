"""
Pin how HeartPrint separates sessions from one another.

Two properties of the distribution shape what a protocol can compare. Two
subjects who attended once have their single sitting filed under both
Session-1 and Session-2, so enrolling on one and probing the other would
compare a recording against itself. Separately, Session-3R and Session-3L are
two labels on the same third visit and legitimately share recordings, so the
same reasoning must not remove those.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_heartprint_dataset


SESSION_FOLDERS = {
    "session1": "Session-1",
    "session2": "Session-2",
    "session3r": "Session-3R",
    "session3l": "Session-3L",
}


def write_recording(root, session, subject, name, seed):
    """
    Write one recording. The seed decides the content, so two calls with the
    same seed produce byte-identical files.
    """
    directory = root / SESSION_FOLDERS[session] / subject
    directory.mkdir(parents=True, exist_ok=True)

    generator = np.random.default_rng(seed)
    samples = generator.normal(size=3747)

    (directory / name).write_text(
        "\n".join(f"{value:.6f}" for value in samples) + "\n"
    )


class HeartPrintDuplicateSessionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        # A subject who attended twice, with different recordings each time.
        write_recording(self.root, "session1", "001", "a.txt", seed=1)
        write_recording(self.root, "session2", "001", "b.txt", seed=2)

        # A subject whose single sitting was filed under both sessions.
        write_recording(self.root, "session1", "119", "same.txt", seed=9)
        write_recording(self.root, "session2", "119", "same.txt", seed=9)

        self.loader = load_heartprint_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["session1"],
        )
        self.loader.dataset_root = self.root

    def test_a_recording_repeated_in_a_later_visit_is_dropped(self):
        recordings = self.loader.load_raw_data()

        self.assertEqual(len(recordings["119"]["session1"]), 1)
        self.assertEqual(len(recordings["119"]["session2"]), 0)

    def test_the_subject_is_kept_in_the_session_that_holds_it_first(self):
        """
        The recording is real; only the second filing of it is not, so the
        subject must survive in Session-1.
        """
        recordings = self.loader.load_raw_data()

        self.assertTrue(recordings["119"]["session1"])

    def test_a_genuine_second_visit_is_untouched(self):
        recordings = self.loader.load_raw_data()

        self.assertEqual(len(recordings["001"]["session1"]), 1)
        self.assertEqual(len(recordings["001"]["session2"]), 1)


class HeartPrintSameVisitTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        write_recording(self.root, "session1", "001", "a.txt", seed=1)

        # One third-visit recording carrying both labels, as the database
        # distributes it.
        write_recording(self.root, "session3r", "001", "visit3.txt", seed=3)
        write_recording(self.root, "session3l", "001", "visit3.txt", seed=3)

    def loader(self, **kwargs):
        loader = load_heartprint_dataset(**kwargs)
        loader.dataset_root = self.root
        return loader

    def test_a_recording_shared_within_one_visit_is_kept_under_both_labels(self):
        """
        Removing it would strip the long-interval protocol of most of its
        cohort while fixing nothing.
        """
        loader = self.loader(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["session1"],
        )
        recordings = loader.load_raw_data()

        self.assertEqual(len(recordings["001"]["session3r"]), 1)
        self.assertEqual(len(recordings["001"]["session3l"]), 1)

    def test_enrolling_on_one_label_and_probing_the_other_is_rejected(self):
        for enrol, probe in (
            ("session3r", "session3l"),
            ("session3l", "session3r"),
        ):
            with self.subTest(enrol=enrol, probe=probe):
                with self.assertRaisesRegex(
                    ValueError,
                    "two labels on the same visit",
                ):
                    load_heartprint_dataset(
                        data_split_mode="cross-session",
                        train_sessions=None,
                        enroll_sessions=[enrol],
                        probe_sessions=[probe],
                    )

    def test_training_on_one_label_and_probing_the_other_is_rejected(self):
        """
        The training session is held to the same rule as the enrolment one:
        probing on recordings the model was trained on measures memorisation.
        """
        for train, probe in (
            ("session3r", "session3l"),
            ("session3l", "session3r"),
        ):
            with self.subTest(train=train, probe=probe):
                with self.assertRaisesRegex(
                    ValueError,
                    "two labels on the same visit",
                ):
                    load_heartprint_dataset(
                        data_split_mode="cross-session",
                        train_sessions=[train],
                        enroll_sessions=None,
                        probe_sessions=[probe],
                    )

    def test_both_labels_may_sit_on_the_enrolment_side(self):
        """
        With the probe drawn from another visit the comparison is sound, and
        training on data that later forms the gallery is a normal pattern.
        """
        load_heartprint_dataset(
            data_split_mode="cross-session",
            train_sessions=["session3r"],
            enroll_sessions=["session3l"],
            probe_sessions=["session1"],
        )

    def test_a_comparison_across_visits_is_accepted(self):
        for enrol, probe in (
            ("session1", "session2"),
            ("session1", "session3l"),
            ("session1", "session3r"),
            ("session2", "session3l"),
        ):
            with self.subTest(enrol=enrol, probe=probe):
                load_heartprint_dataset(
                    data_split_mode="cross-session",
                    train_sessions=None,
                    enroll_sessions=[enrol],
                    probe_sessions=[probe],
                )


class HeartPrintEnumerationTests(unittest.TestCase):
    def test_sessions_are_read_in_a_fixed_order(self):
        """
        The order decides which copy of a repeated recording is kept, so it
        must not depend on the filesystem.
        """
        loader = load_heartprint_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["session1"],
        )

        self.assertEqual(
            loader.SESSION_ORDER,
            ("session1", "session2", "session3r", "session3l"),
        )

    def test_the_third_visit_labels_share_a_visit(self):
        loader = load_heartprint_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["session1"],
        )

        self.assertEqual(
            loader.SESSION_VISIT["session3r"],
            loader.SESSION_VISIT["session3l"],
        )
        self.assertNotEqual(
            loader.SESSION_VISIT["session1"],
            loader.SESSION_VISIT["session2"],
        )

    def test_the_sampling_rate_comes_from_the_configuration(self):
        loader = load_heartprint_dataset(
            data_split_mode="single-session",
            session_for_single_session_evaluation=["session1"],
        )

        self.assertEqual(loader.fs, 250)


if __name__ == "__main__":
    unittest.main()
