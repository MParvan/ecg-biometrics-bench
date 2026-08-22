import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main


class SyntheticContinuousLoader:
    """
    Non-downloading loader used to verify main.py configuration routing.
    """

    def __init__(self):
        self.cfg = {
            "root_dir": "synthetic_continuous",
            "preprocessing": {},
        }
        self.prep_params = {}

        self.train_x = np.zeros(
            (12, 64),
            dtype=np.float32,
        )
        self.train_y = np.asarray(
            [
                f"subject_{index // 2}"
                for index in range(12)
            ]
        )

        self.test_x = np.ones(
            (12, 64),
            dtype=np.float32,
        )
        self.test_y = np.asarray(
            [
                f"subject_{index // 2}"
                for index in range(12)
            ]
        )

        self.load_all_data = Mock(
            side_effect=self._load_all_data
        )

        self.load_session = Mock(
            side_effect=self._load_session
        )

    def _synthetic_provenance(self, labels):
        from load_dataset import _ProvenanceBuilder

        builder = _ProvenanceBuilder()
        for subject in sorted(set(labels.tolist())):
            count = int(np.sum(labels == subject))
            builder.add_block(
                count,
                record_id=f"{subject}_r",
                session_id=f"{subject}_r",
                acquisition_time=None,
                acquisition_order=0,
                source_segment_id=f"{subject}_r#0",
                source_segment_order=0.0,
            )
        return builder.build()

    def _load_all_data(self, return_provenance=False):
        if return_provenance:
            return (
                self.train_x,
                self.train_y,
                self._synthetic_provenance(self.train_y),
            )
        return self.train_x, self.train_y

    def _load_session(self, session_name, return_provenance=False):
        if session_name == "train":
            x, y = self.train_x, self.train_y
        elif session_name == "test":
            x, y = self.test_x, self.test_y
        else:
            raise ValueError(
                f"Unexpected synthetic session: {session_name}"
            )

        if return_provenance:
            return x, y, self._synthetic_provenance(y)
        return x, y



class ContinuousDatasetCLITests(
    unittest.TestCase
):
    def test_mitbih_custom_ranges_reach_loader(self):
        loader = SyntheticContinuousLoader()

        cli_arguments = [
            "main.py",
            "--dataset",
            "mitbih",
            "--task",
            "6",
            "--data_split_mode",
            "custom-split",
            "--train_parts",
            "0",
            "5",
            "--train_parts",
            "10",
            "15",
            "--enroll_parts",
            "15",
            "20",
            "--test_parts",
            "25",
            "30",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--device",
            "cpu",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), patch.object(
            main,
            "load_mitbih_dataset",
            return_value=loader,
        ) as constructor_mock, patch.object(
            main.run,
            "start_experiment_timer",
        ), patch.object(
            main,
            "run_cross_session_verification",
        ) as runner_mock:
            main.main()

        constructor_mock.assert_called_once_with(
            data_split_mode="custom-split",
            num_beats_to_merge=1,
            beat_merge_stride=1,
            preprocessing_config={},
            train_parts=[
                (0.0, 5.0),
                (10.0, 15.0),
            ],
            enrol_parts=[
                (15.0, 20.0),
            ],
            test_parts=[
                (25.0, 30.0),
            ],
            temporal_guard_minutes=0.0,
        )

        self.assertEqual(
            loader.load_session.call_args_list,
            [
                call("train", return_provenance=True),
                call("test", return_provenance=True),
            ],
        )

        runner_mock.assert_called_once()

        positional_arguments = (
            runner_mock.call_args.args
        )

        self.assertIs(
            positional_arguments[0],
            loader.train_x,
        )
        self.assertIs(
            positional_arguments[1],
            loader.train_y,
        )
        self.assertIs(
            positional_arguments[2],
            loader.test_x,
        )
        self.assertIs(
            positional_arguments[3],
            loader.test_y,
        )

    def test_nsrdb_single_segment_range_reaches_loader(self):
        loader = SyntheticContinuousLoader()

        cli_arguments = [
            "main.py",
            "--dataset",
            "nsrdb",
            "--task",
            "1",
            "--data_split_mode",
            "single-segment",
            "--single_segment_range",
            "60",
            "120",
            "--epochs",
            "1",
            "--batch_size",
            "4",
            "--device",
            "cpu",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), patch.object(
            main,
            "load_nsrdb_dataset",
            return_value=loader,
        ) as constructor_mock, patch.object(
            main.run,
            "start_experiment_timer",
        ), patch.object(
            main,
            "run_closed_set_identification",
        ) as runner_mock:
            main.main()

        constructor_mock.assert_called_once_with(
            data_split_mode="single-segment",
            num_beats_to_merge=1,
            beat_merge_stride=1,
            preprocessing_config={},
            single_segment_range=(
                60.0,
                120.0,
            ),
            temporal_guard_minutes=0.0,
        )

        loader.load_all_data.assert_called_once_with(return_provenance=True)
        loader.load_session.assert_not_called()
        runner_mock.assert_called_once()

    def test_invalid_minute_range_is_rejected(self):
        cli_arguments = [
            "main.py",
            "--dataset",
            "mitbih",
            "--task",
            "1",
            "--data_split_mode",
            "single-segment",
            "--single_segment_range",
            "10",
            "5",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), self.assertRaisesRegex(
            SystemExit,
            "2",
        ):
            main.main()

    def test_custom_split_requires_test_parts(self):
        cli_arguments = [
            "main.py",
            "--dataset",
            "mitbih",
            "--task",
            "6",
            "--data_split_mode",
            "custom-split",
            "--train_parts",
            "0",
            "5",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), self.assertRaisesRegex(
            SystemExit,
            "2",
        ):
            main.main()

    def test_cross_session_task_requires_custom_split(self):
        cli_arguments = [
            "main.py",
            "--dataset",
            "nsrdb",
            "--task",
            "6",
            "--data_split_mode",
            "single-segment",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), self.assertRaisesRegex(
            SystemExit,
            "2",
        ):
            main.main()

    def test_minute_ranges_rejected_for_other_datasets(self):
        cli_arguments = [
            "main.py",
            "--dataset",
            "ecgid",
            "--task",
            "1",
            "--data_split_mode",
            "all-available",
            "--train_parts",
            "0",
            "5",
        ]

        with patch.object(
            sys,
            "argv",
            cli_arguments,
        ), self.assertRaisesRegex(
            SystemExit,
            "2",
        ):
            main.main()


if __name__ == "__main__":
    unittest.main()