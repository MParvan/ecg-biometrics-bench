import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

import run


class RuntimeStageProfilingTests(unittest.TestCase):
    def setUp(self):
        run._EXPERIMENT_START_TIME = None
        run._reset_runtime_profile()

    def tearDown(self):
        run._EXPERIMENT_START_TIME = None
        run._reset_runtime_profile()

    def test_timed_call_records_stage_duration_and_result(self):
        with patch(
            "run.torch.cuda.is_available",
            return_value=False,
        ), patch(
            "run.time.perf_counter",
            side_effect=[
                100.0,
                101.0,
                103.5,
                112.0,
            ],
        ):
            run.start_experiment_timer()

            result = run._timed_runtime_call(
                "Synthetic Stage",
                lambda value: value + 1,
                4,
            )

            profile = (
                run._collect_runtime_profile()
            )

        self.assertEqual(
            result,
            5,
        )
        self.assertAlmostEqual(
            profile[
                "Synthetic Stage Time (seconds)"
            ],
            2.5,
        )
        self.assertEqual(
            profile[
                "Synthetic Stage Calls"
            ],
            1,
        )
        self.assertAlmostEqual(
            profile[
                "Total Wall-Clock Time (seconds)"
            ],
            12.0,
        )

    def test_timed_call_records_duration_when_callable_raises(self):
        def failing_function():
            raise RuntimeError(
                "synthetic failure"
            )

        with patch(
            "run.torch.cuda.is_available",
            return_value=False,
        ), patch(
            "run.time.perf_counter",
            side_effect=[
                10.0,
                12.0,
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "synthetic failure",
            ):
                run._timed_runtime_call(
                    "Failed Stage",
                    failing_function,
                )

        self.assertAlmostEqual(
            run._RUNTIME_STAGE_TOTALS[
                "Failed Stage"
            ],
            2.0,
        )
        self.assertEqual(
            run._RUNTIME_STAGE_COUNTS[
                "Failed Stage"
            ],
            1,
        )

    def test_cuda_is_synchronized_at_stage_boundaries(self):
        with patch(
            "run.torch.cuda.is_available",
            return_value=True,
        ), patch(
            "run.torch.cuda.reset_peak_memory_stats",
        ), patch(
            "run.torch.cuda.synchronize",
        ) as synchronize_mock, patch(
            "run.time.perf_counter",
            side_effect=[
                0.0,
                1.0,
                2.0,
            ],
        ):
            run.start_experiment_timer()

            run._timed_runtime_call(
                "CUDA Stage",
                lambda: None,
            )

        self.assertEqual(
            synchronize_mock.call_count,
            2,
        )

    def test_multi_run_profile_contains_individual_and_aggregate_times(self):
        with patch(
            "run.torch.cuda.is_available",
            return_value=False,
        ), patch(
            "run.time.perf_counter",
            side_effect=[
                0.0,
                12.0,
                24.0,
                30.0,
            ],
        ):
            run.start_experiment_timer()

            run._record_multi_run_time(
                run_index=1,
                seed=42,
                started_at=10.0,
            )
            run._record_multi_run_time(
                run_index=2,
                seed=43,
                started_at=20.0,
            )

            profile = (
                run._collect_runtime_profile()
            )

        self.assertEqual(
            profile["Run 1 Seed"],
            42,
        )
        self.assertEqual(
            profile["Run 2 Seed"],
            43,
        )
        self.assertAlmostEqual(
            profile[
                "Run 1 Wall-Clock Time (seconds)"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            profile[
                "Run 2 Wall-Clock Time (seconds)"
            ],
            4.0,
        )
        self.assertAlmostEqual(
            profile[
                "Per-Run Time Mean (seconds)"
            ],
            3.0,
        )
        self.assertAlmostEqual(
            profile[
                "Per-Run Time Std (seconds)"
            ],
            np.std(
                [
                    2.0,
                    4.0,
                ]
            ),
        )
        self.assertAlmostEqual(
            profile[
                "Per-Run Time Min (seconds)"
            ],
            2.0,
        )
        self.assertAlmostEqual(
            profile[
                "Per-Run Time Max (seconds)"
            ],
            4.0,
        )

    def test_starting_new_experiment_resets_previous_stage_data(self):
        run._RUNTIME_STAGE_TOTALS[
            "Old Stage"
        ] = 99.0
        run._RUNTIME_STAGE_COUNTS[
            "Old Stage"
        ] = 3
        run._RUNTIME_RUN_TIMES.append(
            {
                "run_index": 1,
                "seed": 42,
                "seconds": 10.0,
            }
        )

        with patch(
            "run.torch.cuda.is_available",
            return_value=False,
        ), patch(
            "run.time.perf_counter",
            return_value=100.0,
        ):
            run.start_experiment_timer()

        self.assertEqual(
            dict(
                run._RUNTIME_STAGE_TOTALS
            ),
            {},
        )
        self.assertEqual(
            dict(
                run._RUNTIME_STAGE_COUNTS
            ),
            {},
        )
        self.assertEqual(
            run._RUNTIME_RUN_TIMES,
            [],
        )

    def test_all_runner_and_data_cache_paths_are_profiled(self):
        run_source = Path(
            run.__file__
        ).read_text(
            encoding="utf-8"
        )
        main_source = (
            PROJECT_ROOT
            / "main.py"
        ).read_text(
            encoding="utf-8"
        )

        self.assertEqual(
            run_source.count(
                (
                    "partition_stage_started = "
                    "_start_runtime_stage()"
                )
            ),
            8,
        )
        self.assertEqual(
            run_source.count(
                "run_index=i + 1"
            ),
            8,
        )
        self.assertEqual(
            run_source.count(
                "cache.get_weight_cache,"
            ),
            8,
        )
        self.assertEqual(
            run_source.count(
                "cache.save_weight_cache,"
            ),
            8,
        )
        self.assertEqual(
            main_source.count(
                '"Data Cache Read",'
            ),
            2,
        )
        self.assertEqual(
            main_source.count(
                '"Data Cache Write",'
            ),
            2,
        )
        self.assertEqual(
            main_source.count(
                '"Data Preparation (inclusive)",'
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
