import datetime
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import (
    load_cybhi_dataset,
    load_ecgid_dataset,
    load_heartprint_dataset,
    load_mitbih_dataset,
    load_nsrdb_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
)


RECORD_BASED_LOADERS = [
    load_ecgid_dataset,
    load_ptb_dataset,
    load_ptbxl_dataset,
]

ALL_LOADER_CLASSES = [
    load_ecgid_dataset,
    load_heartprint_dataset,
    load_ptb_dataset,
    load_cybhi_dataset,
    load_mitbih_dataset,
    load_nsrdb_dataset,
    load_ptbxl_dataset,
]


def make_record(
    marker,
    recording_date,
    filename,
):
    """
    Create a minimal record dictionary compatible with record-based loaders.
    """
    return {
        "signal": np.asarray(
            [[float(marker)]],
            dtype=np.float32,
        ),
        "fs": 250,
        "date": recording_date,
        "filename": filename,
    }


def make_tagged_signal(marker):
    """
    Create one synthetic signal entry for HeartPrint and CYBHi.
    """
    return {
        "signal": np.asarray(
            [float(marker)],
            dtype=np.float32,
        ),
        "fs": 250,
    }


def synthetic_processed_segments(
    signal,
    sampling_rate,
):
    """
    Replace computational ECG preprocessing with two deterministic segments.
    """
    del sampling_rate

    marker = float(
        np.asarray(signal).reshape(-1)[0]
    )

    return np.full(
        (2, 8),
        fill_value=marker,
        dtype=np.float32,
    )


class DatasetLoaderContractTests(
    unittest.TestCase
):
    def assert_aligned_arrays(
        self,
        samples,
        labels,
    ):
        self.assertIsInstance(
            samples,
            np.ndarray,
        )
        self.assertIsInstance(
            labels,
            np.ndarray,
        )

        self.assertEqual(
            labels.ndim,
            1,
        )

        self.assertEqual(
            samples.shape[0],
            labels.shape[0],
        )

    def test_all_loaders_accept_common_cli_constructor_arguments(
        self,
    ):
        cases = [
            (
                "ecgid",
                load_ecgid_dataset,
                {
                    "data_split_mode": "all-available",
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "heartprint",
                load_heartprint_dataset,
                {
                    "data_split_mode": "single-session",
                    "session_for_single_session_evaluation": [
                        "session1"
                    ],
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "ptb",
                load_ptb_dataset,
                {
                    "data_split_mode": "all-available",
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "cybhi",
                load_cybhi_dataset,
                {
                    "data_split_mode": "single-session",
                    "session_for_single_session_evaluation": [
                        "long-term_S1"
                    ],
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "mitbih",
                load_mitbih_dataset,
                {
                    "data_split_mode": "all-available",
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "nsrdb",
                load_nsrdb_dataset,
                {
                    "data_split_mode": "all-available",
                    "num_beats_to_merge": 2,
                },
            ),
            (
                "ptbxl",
                load_ptbxl_dataset,
                {
                    "data_split_mode": "all-available",
                    "num_beats_to_merge": 2,
                },
            ),
        ]

        for (
            dataset_name,
            loader_class,
            constructor_arguments,
        ) in cases:
            with self.subTest(
                dataset=dataset_name
            ):
                loader = loader_class(
                    **constructor_arguments
                )

                self.assertEqual(
                    loader.data_split_mode,
                    constructor_arguments[
                        "data_split_mode"
                    ],
                )

                self.assertEqual(
                    loader.num_beats,
                    2,
                )

                self.assertIsInstance(
                    loader.cfg,
                    dict,
                )

                self.assertIsInstance(
                    loader.prep_params,
                    dict,
                )

    def test_invalid_split_modes_are_rejected(
        self,
    ):
        for loader_class in ALL_LOADER_CLASSES:
            with self.subTest(
                loader=loader_class.__name__
            ):
                with self.assertRaises(
                    ValueError
                ):
                    loader_class(
                        data_split_mode=(
                            "unsupported-mode"
                        )
                    )

    def test_record_based_cross_session_loaders_drop_incomplete_subjects(
        self,
    ):
        first_date = datetime.datetime(
            2025,
            1,
            1,
        )
        second_date = datetime.datetime(
            2025,
            1,
            2,
        )

        synthetic_records = {
            "complete_subject": [
                make_record(
                    marker=1,
                    recording_date=first_date,
                    filename="record_1",
                ),
                make_record(
                    marker=2,
                    recording_date=second_date,
                    filename="record_2",
                ),
            ],
            "incomplete_subject": [
                make_record(
                    marker=3,
                    recording_date=first_date,
                    filename="record_1",
                ),
            ],
        }

        for loader_class in RECORD_BASED_LOADERS:
            with self.subTest(
                loader=loader_class.__name__
            ):
                loader = loader_class(
                    data_split_mode=(
                        "single-cross-session"
                    )
                )

                loader.load_raw_data = Mock(
                    return_value=synthetic_records
                )

                loader._process_signal = Mock(
                    side_effect=(
                        synthetic_processed_segments
                    )
                )

                train_x, train_y = (
                    loader.load_session(
                        "train"
                    )
                )

                test_x, test_y = (
                    loader.load_session(
                        "test"
                    )
                )

                self.assert_aligned_arrays(
                    train_x,
                    train_y,
                )
                self.assert_aligned_arrays(
                    test_x,
                    test_y,
                )

                np.testing.assert_array_equal(
                    np.unique(train_y),
                    np.asarray(
                        ["complete_subject"]
                    ),
                )

                np.testing.assert_array_equal(
                    np.unique(test_y),
                    np.asarray(
                        ["complete_subject"]
                    ),
                )

                self.assertEqual(
                    len(train_y),
                    2,
                )
                self.assertEqual(
                    len(test_y),
                    2,
                )

                self.assertTrue(
                    np.all(
                        train_x == 1.0
                    )
                )
                self.assertTrue(
                    np.all(
                        test_x == 2.0
                    )
                )

    def test_multi_session_loaders_enforce_subject_intersection(
        self,
    ):
        cases = [
            (
                "heartprint",
                load_heartprint_dataset,
                {
                    "train_sessions": [
                        "Session 1"
                    ],
                    "enroll_sessions": [],
                    "probe_sessions": [
                        "Session-2"
                    ],
                },
                "session1",
                "session2",
            ),
            (
                "cybhi",
                load_cybhi_dataset,
                {
                    "train_sessions": [
                        "long-term_S1"
                    ],
                    "enroll_sessions": [],
                    "probe_sessions": [
                        "long-term_S2"
                    ],
                },
                "long-term_S1",
                "long-term_S2",
            ),
        ]

        for (
            dataset_name,
            loader_class,
            session_arguments,
            normalized_train_session,
            normalized_probe_session,
        ) in cases:
            with self.subTest(
                dataset=dataset_name
            ):
                loader = loader_class(
                    data_split_mode=(
                        "cross-session"
                    ),
                    **session_arguments,
                )

                synthetic_records = {
                    "complete_subject": {
                        normalized_train_session: [
                            make_tagged_signal(
                                1
                            )
                        ],
                        normalized_probe_session: [
                            make_tagged_signal(
                                2
                            )
                        ],
                    },
                    "missing_probe_subject": {
                        normalized_train_session: [
                            make_tagged_signal(
                                3
                            )
                        ],
                    },
                    "missing_train_subject": {
                        normalized_probe_session: [
                            make_tagged_signal(
                                4
                            )
                        ],
                    },
                }

                loader.load_raw_data = Mock(
                    return_value=synthetic_records
                )

                loader._process_signal = Mock(
                    side_effect=(
                        synthetic_processed_segments
                    )
                )

                train_x, train_y = (
                    loader.load_session(
                        "train"
                    )
                )

                test_x, test_y = (
                    loader.load_session(
                        "test"
                    )
                )

                self.assert_aligned_arrays(
                    train_x,
                    train_y,
                )
                self.assert_aligned_arrays(
                    test_x,
                    test_y,
                )

                np.testing.assert_array_equal(
                    np.unique(train_y),
                    np.asarray(
                        ["complete_subject"]
                    ),
                )

                np.testing.assert_array_equal(
                    np.unique(test_y),
                    np.asarray(
                        ["complete_subject"]
                    ),
                )

                self.assertTrue(
                    np.all(
                        train_x == 1.0
                    )
                )

                self.assertTrue(
                    np.all(
                        test_x == 2.0
                    )
                )

    def test_mitbih_custom_split_routes_exact_ranges(
        self,
    ):
        loader = load_mitbih_dataset(
            data_split_mode="custom-split",
            train_parts=[
                (0, 1),
            ],
            enrol_parts=[
                (1, 2),
            ],
            test_parts=[
                (2, 3),
            ],
        )

        raw_signal = np.arange(
            180,
            dtype=np.float32,
        ).reshape(-1, 1)

        loader.load_raw_data = Mock(
            return_value={
                "record_a": {
                    "signal": raw_signal,
                    "fs": 1,
                    "filename": "record_a",
                }
            }
        )

        loader._process_signal = Mock(
            side_effect=lambda signal, fs: (
                np.asarray(
                    [
                        [
                            signal[0, 0],
                            signal[-1, 0],
                        ]
                    ],
                    dtype=np.float32,
                )
            )
        )

        train_x, train_y = (
            loader.load_session(
                "train"
            )
        )

        enrol_x, enrol_y = (
            loader.load_session(
                "enrollment"
            )
        )

        test_x, test_y = (
            loader.load_session(
                "probe"
            )
        )

        self.assert_aligned_arrays(
            train_x,
            train_y,
        )
        self.assert_aligned_arrays(
            enrol_x,
            enrol_y,
        )
        self.assert_aligned_arrays(
            test_x,
            test_y,
        )

        np.testing.assert_array_equal(
            train_x,
            np.asarray(
                [[0.0, 59.0]],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            enrol_x,
            np.asarray(
                [[60.0, 119.0]],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            test_x,
            np.asarray(
                [[120.0, 179.0]],
                dtype=np.float32,
            ),
        )

        np.testing.assert_array_equal(
            train_y,
            np.asarray(
                ["record_a"]
            ),
        )
        np.testing.assert_array_equal(
            enrol_y,
            np.asarray(
                ["record_a"]
            ),
        )
        np.testing.assert_array_equal(
            test_y,
            np.asarray(
                ["record_a"]
            ),
        )

    def test_nsrdb_custom_split_requests_exact_ranges(
        self,
    ):
        loader = load_nsrdb_dataset(
            data_split_mode="custom-split",
            train_parts=[
                (0, 60),
            ],
            enrol_parts=[
                (60, 120),
            ],
            test_parts=[
                (120, 180),
            ],
        )

        def synthetic_slice_loader(
            min_ranges=None,
        ):
            marker = float(
                min_ranges[0][0]
            )

            return {
                "record_a": [
                    {
                        "signal": np.full(
                            (10, 1),
                            fill_value=marker,
                            dtype=np.float32,
                        ),
                        "fs": 1,
                    }
                ]
            }

        loader.load_raw_data_slices = Mock(
            side_effect=synthetic_slice_loader
        )

        loader._process_signal = Mock(
            side_effect=lambda signal, fs: (
                np.asarray(
                    [
                        [
                            signal[0, 0],
                            signal[-1, 0],
                        ]
                    ],
                    dtype=np.float32,
                )
            )
        )

        train_x, train_y = (
            loader.load_session(
                "train"
            )
        )

        enrol_x, enrol_y = (
            loader.load_session(
                "enrollment"
            )
        )

        test_x, test_y = (
            loader.load_session(
                "probe"
            )
        )

        self.assertEqual(
            loader.load_raw_data_slices
            .call_args_list,
            [
                call(
                    min_ranges=[
                        (0, 60)
                    ]
                ),
                call(
                    min_ranges=[
                        (60, 120)
                    ]
                ),
                call(
                    min_ranges=[
                        (120, 180)
                    ]
                ),
            ],
        )

        self.assert_aligned_arrays(
            train_x,
            train_y,
        )
        self.assert_aligned_arrays(
            enrol_x,
            enrol_y,
        )
        self.assert_aligned_arrays(
            test_x,
            test_y,
        )

        self.assertTrue(
            np.all(
                train_x == 0.0
            )
        )
        self.assertTrue(
            np.all(
                enrol_x == 60.0
            )
        )
        self.assertTrue(
            np.all(
                test_x == 120.0
            )
        )

        np.testing.assert_array_equal(
            train_y,
            np.asarray(
                ["record_a"]
            ),
        )
        np.testing.assert_array_equal(
            enrol_y,
            np.asarray(
                ["record_a"]
            ),
        )
        np.testing.assert_array_equal(
            test_y,
            np.asarray(
                ["record_a"]
            ),
        )

    def test_nsrdb_single_segment_requests_only_selected_range(
        self,
    ):
        loader = load_nsrdb_dataset(
            data_split_mode="single-segment",
            single_segment_range=(
                30,
                45,
            ),
        )

        loader.load_raw_data_slices = Mock(
            return_value={}
        )

        samples, labels = (
            loader.load_all_data()
        )

        loader.load_raw_data_slices.assert_called_once_with(
            min_ranges=[
                (
                    30,
                    45,
                )
            ]
        )

        self.assert_aligned_arrays(
            samples,
            labels,
        )

        self.assertEqual(
            samples.shape,
            (0, 0),
        )
        self.assertEqual(
            labels.shape,
            (0,),
        )

    def test_all_loaders_return_aligned_empty_arrays(
        self,
    ):
        cases = [
            (
                "ecgid",
                load_ecgid_dataset(
                    data_split_mode=(
                        "all-available"
                    )
                ),
                "load_raw_data",
            ),
            (
                "heartprint",
                load_heartprint_dataset(
                    data_split_mode=(
                        "single-session"
                    ),
                    session_for_single_session_evaluation=[
                        "session1"
                    ],
                ),
                "load_raw_data",
            ),
            (
                "ptb",
                load_ptb_dataset(
                    data_split_mode=(
                        "all-available"
                    )
                ),
                "load_raw_data",
            ),
            (
                "cybhi",
                load_cybhi_dataset(
                    data_split_mode=(
                        "single-session"
                    ),
                    session_for_single_session_evaluation=[
                        "long-term_S1"
                    ],
                ),
                "load_raw_data",
            ),
            (
                "mitbih",
                load_mitbih_dataset(
                    data_split_mode=(
                        "single-segment"
                    ),
                    single_segment_range=(
                        0,
                        1,
                    ),
                ),
                "load_raw_data",
            ),
            (
                "nsrdb",
                load_nsrdb_dataset(
                    data_split_mode=(
                        "single-segment"
                    ),
                    single_segment_range=(
                        0,
                        1,
                    ),
                ),
                "load_raw_data_slices",
            ),
            (
                "ptbxl",
                load_ptbxl_dataset(
                    data_split_mode=(
                        "all-available"
                    )
                ),
                "load_raw_data",
            ),
        ]

        for (
            dataset_name,
            loader,
            raw_loader_name,
        ) in cases:
            with self.subTest(
                dataset=dataset_name
            ):
                setattr(
                    loader,
                    raw_loader_name,
                    Mock(
                        return_value={}
                    ),
                )

                samples, labels = (
                    loader.load_all_data()
                )

                self.assert_aligned_arrays(
                    samples,
                    labels,
                )

                self.assertEqual(
                    samples.shape,
                    (0, 0),
                )

                self.assertEqual(
                    labels.shape,
                    (0,),
                )


if __name__ == "__main__":
    unittest.main()