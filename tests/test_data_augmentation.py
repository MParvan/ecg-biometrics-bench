import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data_augmentation import ECGAugmentation


def make_ecg_batch(
    number_of_samples=4,
    signal_length=128,
):
    time_axis = np.linspace(
        0.0,
        2.0 * np.pi,
        signal_length,
        endpoint=False,
    )

    signals = []

    for sample_index in range(
        number_of_samples
    ):
        signal = (
            np.sin(
                (
                    1.0
                    + 0.1 * sample_index
                )
                * time_axis
            )
            + 0.2
            * np.cos(
                2.0 * time_axis
            )
            + 0.05 * sample_index
        )

        signals.append(
            signal
        )

    return np.asarray(
        signals,
        dtype=np.float32,
    )


class ECGAugmentationTests(
    unittest.TestCase
):
    def setUp(self):
        self.signals = make_ecg_batch()

    def test_all_supported_methods_preserve_shape_and_dtype(
        self,
    ):
        method_cases = [
            (
                "gaussian",
                {
                    "std": 0.02,
                },
            ),
            (
                "amplitude",
                {
                    "scale_range": (
                        0.9,
                        1.1,
                    ),
                },
            ),
            (
                "timeshift",
                {
                    "max_shift": 5,
                },
            ),
            (
                "baseline_wander",
                {
                    "freq": 0.3,
                    "amp": 0.05,
                    "fs": 250,
                },
            ),
            (
                "time_warp",
                {
                    "sigma": 0.1,
                    "num_knots": 4,
                },
            ),
            (
                "cutout",
                {
                    "num_holes": 2,
                    "length": 10,
                },
            ),
            (
                "emg_noise",
                {
                    "fs": 250,
                    "std": 0.03,
                },
            ),
            (
                "istft_augment",
                {
                    "fs": 250,
                    "nperseg": 64,
                    "noverlap": 32,
                    "noise_std": 0.02,
                },
            ),
        ]

        for method, parameters in method_cases:
            with self.subTest(
                method=method
            ):
                augmenter = ECGAugmentation(
                    seed=42
                )

                augmented = augmenter.apply(
                    self.signals,
                    method,
                    **parameters,
                )

                self.assertEqual(
                    augmented.shape,
                    self.signals.shape,
                )

                self.assertEqual(
                    augmented.dtype,
                    np.float32,
                )

                self.assertTrue(
                    np.isfinite(
                        augmented
                    ).all()
                )

    def test_same_seed_produces_identical_augmentation(
        self,
    ):
        first_augmenter = (
            ECGAugmentation(
                seed=123
            )
        )

        second_augmenter = (
            ECGAugmentation(
                seed=123
            )
        )

        first_output = (
            first_augmenter.gaussian(
                self.signals,
                std=0.05,
            )
        )

        second_output = (
            second_augmenter.gaussian(
                self.signals,
                std=0.05,
            )
        )

        np.testing.assert_array_equal(
            first_output,
            second_output,
        )

    def test_different_seeds_change_augmentation(
        self,
    ):
        first_output = (
            ECGAugmentation(
                seed=1
            ).gaussian(
                self.signals,
                std=0.05,
            )
        )

        second_output = (
            ECGAugmentation(
                seed=2
            ).gaussian(
                self.signals,
                std=0.05,
            )
        )

        self.assertFalse(
            np.array_equal(
                first_output,
                second_output,
            )
        )

    def test_augmentation_does_not_modify_input_array(
        self,
    ):
        original = self.signals.copy()

        augmenter = ECGAugmentation(
            seed=42
        )

        augmenter.cutout(
            self.signals,
            num_holes=2,
            length=20,
        )

        np.testing.assert_array_equal(
            self.signals,
            original,
        )

    def test_cutout_can_mask_complete_signal(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        augmented = augmenter.cutout(
            self.signals,
            num_holes=1,
            length=self.signals.shape[1],
        )

        np.testing.assert_array_equal(
            augmented,
            np.zeros_like(
                self.signals
            ),
        )

    def test_zero_intensity_operations_preserve_signal(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        np.testing.assert_array_equal(
            augmenter.gaussian(
                self.signals,
                std=0.0,
            ),
            self.signals,
        )

        np.testing.assert_array_equal(
            augmenter.timeshift(
                self.signals,
                max_shift=0,
            ),
            self.signals,
        )

        np.testing.assert_array_equal(
            augmenter.emg_noise(
                self.signals,
                fs=250,
                std=0.0,
            ),
            self.signals,
        )

        np.testing.assert_array_equal(
            augmenter.cutout(
                self.signals,
                num_holes=0,
                length=10,
            ),
            self.signals,
        )

    def test_one_dimensional_signal_becomes_single_item_batch(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        augmented = augmenter.gaussian(
            self.signals[0],
            std=0.01,
        )

        self.assertEqual(
            augmented.shape,
            (
                1,
                self.signals.shape[1],
            ),
        )

    def test_empty_batch_is_preserved(
        self,
    ):
        empty_batch = np.empty(
            (
                0,
                self.signals.shape[1],
            ),
            dtype=np.float32,
        )

        augmenter = ECGAugmentation(
            seed=42
        )

        for method in (
            ECGAugmentation.SUPPORTED_METHODS
        ):
            with self.subTest(
                method=method
            ):
                parameters = {}

                if method in {
                    "baseline_wander",
                    "emg_noise",
                    "istft_augment",
                }:
                    parameters["fs"] = 250

                output = augmenter.apply(
                    empty_batch,
                    method,
                    **parameters,
                )

                self.assertEqual(
                    output.shape,
                    empty_batch.shape,
                )

                self.assertEqual(
                    output.dtype,
                    np.float32,
                )

    def test_unknown_method_is_rejected(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown augmentation method",
        ):
            augmenter.apply(
                self.signals,
                "unsupported",
            )

    def test_invalid_parameters_are_rejected(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        invalid_calls = [
            (
                lambda: augmenter.gaussian(
                    self.signals,
                    std=-0.1,
                ),
                "std",
            ),
            (
                lambda: augmenter.amplitude(
                    self.signals,
                    scale_range=(
                        1.2,
                        0.8,
                    ),
                ),
                "scale_range",
            ),
            (
                lambda: augmenter.timeshift(
                    self.signals,
                    max_shift=-1,
                ),
                "max_shift",
            ),
            (
                lambda: augmenter.cutout(
                    self.signals,
                    length=(
                        self.signals.shape[1]
                        + 1
                    ),
                ),
                "signal length",
            ),
            (
                lambda: augmenter.emg_noise(
                    self.signals,
                    fs=40,
                ),
                "greater than 40",
            ),
            (
                lambda: augmenter.istft_augment(
                    self.signals,
                    fs=250,
                    nperseg=32,
                    noverlap=32,
                ),
                "smaller than nperseg",
            ),
        ]

        for invalid_call, message in invalid_calls:
            with self.subTest(
                message=message
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    invalid_call()

    def test_method_aliases_are_supported(
        self,
    ):
        augmenter = ECGAugmentation(
            seed=42
        )

        alias_cases = [
            (
                "time-shift",
                {
                    "max_shift": 2,
                },
            ),
            (
                "baseline",
                {
                    "fs": 250,
                },
            ),
            (
                "warp",
                {},
            ),
            (
                "emg",
                {
                    "fs": 250,
                },
            ),
            (
                "istft",
                {
                    "fs": 250,
                    "nperseg": 64,
                    "noverlap": 32,
                },
            ),
        ]

        for method, parameters in alias_cases:
            with self.subTest(
                method=method
            ):
                output = augmenter.apply(
                    self.signals,
                    method,
                    **parameters,
                )

                self.assertEqual(
                    output.shape,
                    self.signals.shape,
                )

                self.assertTrue(
                    np.isfinite(
                        output
                    ).all()
                )


if __name__ == "__main__":
    unittest.main()