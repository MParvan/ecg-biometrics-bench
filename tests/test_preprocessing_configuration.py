import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

for optional_module in [
    "wfdb",
    "patoolib",
    "neurokit2",
]:
    if importlib.util.find_spec(optional_module) is None:
        sys.modules[optional_module] = types.ModuleType(optional_module)

import load_dataset
from filtering import Filtering
from preprocessing import Preprocessing


LOADER_CLASSES = [
    load_dataset.load_ecgid_dataset,
    load_dataset.load_heartprint_dataset,
    load_dataset.load_ptb_dataset,
    load_dataset.load_cybhi_dataset,
    load_dataset.load_mitbih_dataset,
    load_dataset.load_nsrdb_dataset,
    load_dataset.load_ptbxl_dataset,
]


class RecordingPreprocessor:
    def __init__(self):
        self.kwargs = None

    def preprocess_ecg(self, **kwargs):
        self.kwargs = kwargs
        return np.asarray([[1.0, 2.0]], dtype=np.float32)


class PreprocessingConfigurationTests(unittest.TestCase):
    def test_legacy_configuration_preserves_submitted_protocol(self):
        config = load_dataset._normalize_preprocessing_config(
            {
                "pre_s": 0.2,
                "post_s": 0.4,
                "bandpass": True,
                "lowcut": 0.5,
                "highcut": 40.0,
                "normalize": True,
            }
        )

        self.assertEqual(config["mode"], "beat")
        self.assertEqual(config["rpeak_method"], "pantompkins")
        self.assertTrue(config["align_peak"])
        self.assertEqual(config["filter_method"], "butter")
        self.assertEqual(
            config["filter_parameters"],
            {
                "low": 0.5,
                "high": 40.0,
                "order": 4,
            },
        )
        self.assertEqual(config["normalization_method"], "zscore")

    def test_legacy_window_aliases_are_not_silently_ignored(self):
        config = load_dataset._normalize_preprocessing_config(
            {
                "mode": "blind",
                "window_len": 7.5,
                "stride": 2.5,
            }
        )

        self.assertEqual(config["window_s"], 7.5)
        self.assertEqual(config["stride_s"], 2.5)

    def test_canonical_overrides_merge_filter_parameters(self):
        config = load_dataset._normalize_preprocessing_config(
            {
                "filter_parameters": {
                    "low": 0.75,
                }
            },
            {
                "filter_parameters": {
                    "high": 35.0,
                    "order": 6,
                },
                "normalization_method": "minmax",
            },
        )

        self.assertEqual(
            config["filter_parameters"],
            {
                "low": 0.75,
                "high": 35.0,
                "order": 6,
            },
        )
        self.assertEqual(config["normalization_method"], "minmax")

    def test_changing_filter_method_drops_inactive_butter_parameters(self):
        config = load_dataset._normalize_preprocessing_config(
            {
                "filter_method": "notch",
                "filter_parameters": {
                    "freq": 50.0,
                    "quality": 30.0,
                },
            }
        )

        self.assertEqual(config["filter_method"], "notch")
        self.assertEqual(
            config["filter_parameters"],
            {
                "freq": 50.0,
                "quality": 30.0,
            },
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown preprocessing parameter",
        ):
            load_dataset._normalize_preprocessing_config(
                {
                    "window_seconds_typo": 5.0,
                }
            )

        with self.assertRaisesRegex(
            ValueError,
            "positive integer",
        ):
            load_dataset._normalize_preprocessing_config(
                {
                    "filter_parameters": {
                        "order": 0,
                    }
                }
            )

    def test_shared_preprocessing_call_forwards_complete_mapping(self):
        recorder = RecordingPreprocessor()
        config = load_dataset._normalize_preprocessing_config(
            {
                "rpeak_method": "hamilton",
                "align_peak": False,
                "filter_parameters": {
                    "low": 1.0,
                    "high": 30.0,
                    "order": 5,
                },
            }
        )

        output = load_dataset._preprocess_signal(
            recorder,
            np.arange(20, dtype=np.float32),
            250,
            config,
        )

        np.testing.assert_array_equal(
            output,
            np.asarray([[1.0, 2.0]], dtype=np.float32),
        )
        self.assertEqual(recorder.kwargs["rpeak_method"], "hamilton")
        self.assertFalse(recorder.kwargs["align_peak"])
        self.assertEqual(
            recorder.kwargs["filter_kwargs"],
            {
                "low": 1.0,
                "high": 30.0,
                "order": 5,
            },
        )

    def test_preprocessing_forwards_detector_alignment_and_filter_order(self):
        preprocessor = Preprocessing()
        signal = np.sin(
            np.linspace(0.0, 8.0 * np.pi, 200)
        ).astype(np.float32)

        with mock.patch.object(
            preprocessor.filtering,
            "butter",
            return_value=signal,
        ) as butter_mock, mock.patch.object(
            preprocessor,
            "detect_r_peaks",
            return_value=np.asarray([60, 120]),
        ) as detector_mock, mock.patch.object(
            preprocessor,
            "cut_beats",
            return_value=[signal[40:100], signal[100:160]],
        ) as cut_mock:
            result = preprocessor.preprocess_ecg(
                signal,
                fs=250,
                mode="beat",
                pre_s=0.2,
                post_s=0.4,
                rpeak_method="hamilton",
                align_peak=False,
                filter_method="butter",
                filter_kwargs={
                    "low": 0.5,
                    "high": 40.0,
                    "order": 4,
                },
                norm_method=None,
            )

        self.assertEqual(result.shape, (2, 60))
        butter_mock.assert_called_once_with(
            signal,
            250.0,
            low=0.5,
            high=40.0,
            order=4,
        )
        detector_mock.assert_called_once_with(
            signal,
            250.0,
            method="hamilton",
        )
        self.assertFalse(cut_mock.call_args.kwargs["align_peak"])

    def test_all_loaders_accept_and_record_canonical_overrides(self):
        for loader_class in LOADER_CLASSES:
            with self.subTest(loader=loader_class.__name__):
                self.assertIn(
                    "preprocessing_config",
                    inspect.signature(loader_class).parameters,
                )

                loader = loader_class(
                    preprocessing_config={
                        "filter_parameters": {
                            "order": 6,
                        },
                        "rpeak_method": "hamilton",
                    }
                )

                self.assertEqual(
                    loader.prep_params["filter_parameters"]["order"],
                    6,
                )
                self.assertEqual(
                    loader.prep_params["rpeak_method"],
                    "hamilton",
                )

    def test_filtering_default_remains_fourth_order(self):
        self.assertEqual(
            inspect.signature(Filtering.butter)
            .parameters["order"]
            .default,
            4,
        )

    def test_repository_configs_record_fourth_order_filter(self):
        config_path = PROJECT_ROOT / "config.yaml"
        config = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )

        for dataset_name, dataset_config in config["datasets"].items():
            with self.subTest(dataset=dataset_name):
                preprocessing = dataset_config["preprocessing"]
                self.assertEqual(preprocessing["mode"], "beat")
                self.assertEqual(
                    preprocessing["rpeak_method"],
                    "pantompkins",
                )
                self.assertTrue(preprocessing["align_peak"])
                self.assertEqual(
                    preprocessing["filter_parameters"]["order"],
                    4,
                )

    def test_main_exposes_and_propagates_preprocessing_mapping(self):
        syntax_tree = ast.parse(
            (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        )

        string_values = {
            node.value
            for node in ast.walk(syntax_tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        }

        self.assertIn("--preprocessing_parameters", string_values)
        self.assertIn("preprocessing_config", string_values)



if __name__ == "__main__":
    unittest.main()
