import datetime
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_ecgid_dataset


def make_record(value, date, filename):
    return {
        "signal": np.full(8, value, dtype=np.float32),
        "date": date,
        "fs": 250,
        "filename": filename,
    }


class ECGIDRegimeTests(unittest.TestCase):
    def setUp(self):
        day_1 = datetime.date(2024, 1, 1)

        self.synthetic_data = {
            # This subject is not eligible for any protocol requiring
            # two separate records.
            "single_record_subject": [
                make_record(1.0, day_1, "rec_1.hea"),
            ],

            # This subject is eligible because it has two separate records.
            "eligible_subject": [
                make_record(2.0, day_1, "rec_1.hea"),
                make_record(3.0, day_1, "rec_2.hea"),
            ],
        }

    @staticmethod
    def fake_process_signal(signal, fs):
        # Convert each recording into one synthetic segment.
        return np.asarray(signal, dtype=np.float32).reshape(1, -1)

    def assert_single_record_subject_is_dropped(self, mode):
        loader = load_ecgid_dataset(data_split_mode=mode)

        with patch.object(
            loader,
            "load_raw_data",
            return_value=self.synthetic_data,
        ), patch.object(
            loader,
            "_process_signal",
            side_effect=self.fake_process_signal,
        ):
            _, train_labels = loader.load_session("train")
            _, test_labels = loader.load_session("test")

        self.assertEqual(set(train_labels.tolist()), {"eligible_subject"})
        self.assertEqual(set(test_labels.tolist()), {"eligible_subject"})

    def test_single_cross_session_requires_two_records(self):
        self.assert_single_record_subject_is_dropped(
            "single-cross-session"
        )

    def test_single_shot_short_term_requires_two_day_one_records(self):
        self.assert_single_record_subject_is_dropped(
            "single-shot-short-term"
        )

    def test_leave_last_out_short_term_requires_two_day_one_records(self):
        self.assert_single_record_subject_is_dropped(
            "leave-last-out-short-term"
        )


if __name__ == "__main__":
    unittest.main()