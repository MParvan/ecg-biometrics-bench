import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run


def build_per_run_results(values_by_metric):
    """
    Build a per-run record list from metric name to per-seed values.
    """
    metric_names = list(values_by_metric)
    run_count = len(values_by_metric[metric_names[0]])

    return [
        {
            "run_index": index + 1,
            "seed": 42 + index,
            "metrics": {
                name: values_by_metric[name][index]
                for name in metric_names
            },
        }
        for index in range(run_count)
    ]


EER_VALUES = [0.0565, 0.0601, 0.0498, 0.0543, 0.0577]


class MetricUncertaintyTests(unittest.TestCase):
    """
    Interval arithmetic must match the textbook definitions exactly.
    """

    def test_t_interval_matches_the_closed_form(self):
        summary = run._summarize_metric_uncertainty(
            EER_VALUES
        )

        values = np.asarray(EER_VALUES)
        expected_mean = float(np.mean(values))
        expected_sample_std = float(
            np.std(values, ddof=1)
        )
        expected_error = expected_sample_std / np.sqrt(
            len(values)
        )
        expected_critical = stats.t.ppf(
            0.975,
            df=len(values) - 1,
        )

        self.assertAlmostEqual(
            summary["mean"],
            expected_mean,
        )
        self.assertAlmostEqual(
            summary["sample_std"],
            expected_sample_std,
        )
        self.assertAlmostEqual(
            summary["t_interval"]["lower"],
            expected_mean - expected_critical * expected_error,
        )
        self.assertAlmostEqual(
            summary["t_interval"]["upper"],
            expected_mean + expected_critical * expected_error,
        )

    def test_population_std_matches_the_reported_aggregate(self):
        # The aggregate "mean +/- std" uses the population form, so the
        # summary must expose exactly that value for cross-checking.
        summary = run._summarize_metric_uncertainty(
            EER_VALUES
        )

        self.assertAlmostEqual(
            summary["population_std"],
            float(np.std(np.asarray(EER_VALUES))),
        )

    def test_interval_brackets_the_mean(self):
        summary = run._summarize_metric_uncertainty(
            EER_VALUES
        )

        for interval_name in (
            "t_interval",
            "bootstrap_interval",
        ):
            with self.subTest(interval=interval_name):
                interval = summary[interval_name]

                self.assertLessEqual(
                    interval["lower"],
                    summary["mean"],
                )
                self.assertGreaterEqual(
                    interval["upper"],
                    summary["mean"],
                )

    def test_bootstrap_is_deterministic(self):
        first = run._summarize_metric_uncertainty(
            EER_VALUES
        )
        second = run._summarize_metric_uncertainty(
            EER_VALUES
        )

        self.assertEqual(
            first["bootstrap_interval"],
            second["bootstrap_interval"],
        )

    def test_wider_confidence_level_widens_the_interval(self):
        narrow = run._summarize_metric_uncertainty(
            EER_VALUES,
            confidence_level=0.80,
        )
        wide = run._summarize_metric_uncertainty(
            EER_VALUES,
            confidence_level=0.99,
        )

        self.assertLess(
            narrow["t_interval"]["upper"]
            - narrow["t_interval"]["lower"],
            wide["t_interval"]["upper"]
            - wide["t_interval"]["lower"],
        )

    def test_single_run_reports_no_interval(self):
        summary = run._summarize_metric_uncertainty(
            [0.05]
        )

        self.assertEqual(summary["runs"], 1)
        self.assertIsNone(summary["sample_std"])
        self.assertIsNone(summary["t_interval"])
        self.assertIsNone(summary["bootstrap_interval"])

    def test_identical_values_produce_a_zero_width_interval(self):
        summary = run._summarize_metric_uncertainty(
            [0.05] * 5
        )

        self.assertEqual(
            summary["t_interval"]["lower"],
            summary["t_interval"]["upper"],
        )

    def test_empty_values_return_none(self):
        self.assertIsNone(
            run._summarize_metric_uncertainty([])
        )

    def test_non_finite_values_return_none(self):
        self.assertIsNone(
            run._summarize_metric_uncertainty(
                [0.05, float("nan")]
            )
        )

    def test_invalid_confidence_level_is_rejected(self):
        with self.assertRaises(ValueError):
            run._summarize_metric_uncertainty(
                EER_VALUES,
                confidence_level=1.0,
            )


class PerRunSummaryTests(unittest.TestCase):
    """
    The record-level summary covers every metric the runs reported.
    """

    def test_every_metric_is_summarized(self):
        per_run_results = build_per_run_results(
            {
                "EER": EER_VALUES,
                "AUC": [0.97, 0.971, 0.969, 0.972, 0.968],
                "d-prime": [3.2, 3.3, 3.1, 3.25, 3.15],
                "TAR@0.1%FAR": [0.78, 0.79, 0.77, 0.80, 0.76],
            }
        )

        summary = run._summarize_per_run_uncertainty(
            per_run_results
        )

        self.assertEqual(summary["runs"], 5)
        self.assertEqual(
            set(summary["metrics"]),
            {
                "EER",
                "AUC",
                "d-prime",
                "TAR@0.1%FAR",
            },
        )

    def test_empty_input_returns_none(self):
        self.assertIsNone(
            run._summarize_per_run_uncertainty([])
        )
        self.assertIsNone(
            run._summarize_per_run_uncertainty(None)
        )

    def test_records_without_metrics_return_none(self):
        self.assertIsNone(
            run._summarize_per_run_uncertainty(
                [{"run_index": 1, "seed": 42}]
            )
        )

    def test_summary_does_not_mutate_the_input(self):
        per_run_results = build_per_run_results(
            {"EER": EER_VALUES}
        )
        before = [
            dict(record["metrics"])
            for record in per_run_results
        ]

        run._summarize_per_run_uncertainty(
            per_run_results
        )

        after = [
            dict(record["metrics"])
            for record in per_run_results
        ]

        self.assertEqual(before, after)


class StructuredRecordTests(unittest.TestCase):
    """
    Intervals are added to the record without disturbing existing fields.
    """

    def _build_record(self, per_run_results):
        import datetime

        return run._build_structured_experiment_record(
            experiment_time=datetime.datetime(2026, 8, 2, 12, 0, 0),
            task_name="Closed-Set Verification",
            dataset_name="ecgid",
            metrics_dict={
                "EER": "0.0557 ± 0.0035",
            },
            data_stats={},
            hyperparams={},
            dataset_kwargs={},
            software_environment={},
            source_revision={},
            runtime_profile={},
            per_run_results=per_run_results,
        )

    def test_record_contains_the_uncertainty_block(self):
        record = self._build_record(
            build_per_run_results({"EER": EER_VALUES})
        )

        self.assertIn(
            "across_seed_uncertainty",
            record,
        )
        self.assertIn(
            "EER",
            record["across_seed_uncertainty"]["metrics"],
        )

    def test_aggregate_results_are_untouched(self):
        record = self._build_record(
            build_per_run_results({"EER": EER_VALUES})
        )

        self.assertEqual(
            record["results"]["EER"]["display"],
            "0.0557 ± 0.0035",
        )

    def test_per_run_results_are_untouched(self):
        per_run_results = build_per_run_results(
            {"EER": EER_VALUES}
        )
        record = self._build_record(per_run_results)

        self.assertEqual(
            len(record["per_run_results"]),
            5,
        )

    def test_single_run_record_has_no_uncertainty_block(self):
        record = self._build_record(None)

        self.assertIsNone(
            record["across_seed_uncertainty"]
        )


if __name__ == "__main__":
    unittest.main()


class AggregateMetricParsingTests(unittest.TestCase):
    """
    Aggregate 'mean +/- std' strings must become numeric structured values.

    The task runners format multi-run metrics with U+00B1. If the parser
    does not recognise that character, every structured record stores an
    opaque string and downstream analysis silently loses the numbers.
    """

    def test_runner_separator_is_parsed(self):
        parsed = run._to_structured_result_value(
            "0.9500 \u00b1 0.0100"
        )

        self.assertEqual(parsed["mean"], 0.95)
        self.assertEqual(parsed["std"], 0.01)
        self.assertEqual(
            parsed["display"],
            "0.9500 \u00b1 0.0100",
        )

    def test_ascii_separator_is_parsed(self):
        parsed = run._to_structured_result_value(
            "0.9500 +/- 0.0100"
        )

        self.assertEqual(parsed["mean"], 0.95)
        self.assertEqual(parsed["std"], 0.01)

    def test_runners_emit_a_recognised_separator(self):
        # Guards against an encoding round-trip silently replacing the
        # separator in either the runners or the parser.
        import io
        from pathlib import Path

        source = io.open(
            Path(run.__file__),
            encoding="utf-8",
        ).read()

        self.assertIn(
            "{r1_mean:.4f} \u00b1 {r1_std:.4f}",
            source,
        )
        self.assertIn(
            "\u00b1",
            run._AGGREGATE_METRIC_SEPARATORS,
        )

    def test_plain_numbers_pass_through(self):
        self.assertEqual(
            run._to_structured_result_value(0.95),
            0.95,
        )

    def test_unrelated_strings_pass_through(self):
        self.assertEqual(
            run._to_structured_result_value("n/a"),
            "n/a",
        )

    def test_ambiguous_string_is_not_parsed(self):
        self.assertEqual(
            run._to_structured_result_value(
                "0.9 \u00b1 0.1 \u00b1 0.2"
            ),
            "0.9 \u00b1 0.1 \u00b1 0.2",
        )
