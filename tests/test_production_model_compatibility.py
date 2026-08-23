import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run
from models import (
    DeepECG,
    ECGTransformer,
    HybridCNNLSTM,
    RNN_ECG,
    ResNet1D,
)


PRODUCTION_MODELS = [
    DeepECG,
    ResNet1D,
    RNN_ECG,
    HybridCNNLSTM,
    ECGTransformer,
]


def make_synthetic_ecg_dataset(
    number_of_subjects=5,
    samples_per_subject=8,
    signal_length=64,
    seed=2026,
):
    """
    Create deterministic ECG-like samples for model integration testing.

    Each subject has a distinct waveform and small sample-level variations.
    """
    random_generator = np.random.default_rng(
        seed
    )

    time_axis = np.linspace(
        0.0,
        2.0 * np.pi,
        signal_length,
        endpoint=False,
    )

    samples = []
    labels = []

    for subject_index in range(
        number_of_subjects
    ):
        base_waveform = (
            np.sin(
                (
                    1.0
                    + 0.20 * subject_index
                )
                * time_axis
            )
            + 0.25
            * np.cos(
                (
                    2.0
                    + 0.10 * subject_index
                )
                * time_axis
            )
            + 0.05 * subject_index
        )

        for _ in range(samples_per_subject):
            noise = random_generator.normal(
                loc=0.0,
                scale=0.04,
                size=signal_length,
            )

            samples.append(
                (
                    base_waveform + noise
                ).astype(np.float32)
            )

            labels.append(
                f"subject_{subject_index}"
            )

    return (
        np.stack(samples),
        np.asarray(labels),
    )


class ProductionModelCompatibilityTests(
    unittest.TestCase
):
    @classmethod
    def setUpClass(cls):
        cls.previous_thread_count = (
            torch.get_num_threads()
        )

        # Small synthetic workloads are faster and more deterministic
        # without excessive CPU thread scheduling.
        torch.set_num_threads(1)

        cls.synthetic_x, cls.synthetic_y = (
            make_synthetic_ecg_dataset()
        )

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(
            cls.previous_thread_count
        )

    def test_classifier_and_embedding_interfaces(self):
        """
        Every CLI model must provide logits and embeddings through the
        include_top interface used throughout run.py.
        """
        batch_size = 4
        input_channels = 2
        signal_length = 64
        number_of_classes = 5

        input_tensor = torch.randn(
            batch_size,
            input_channels,
            signal_length,
        )

        target_labels = torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.long,
        )

        for model_class in PRODUCTION_MODELS:
            with self.subTest(
                model=model_class.__name__
            ):
                torch.manual_seed(42)

                model = model_class(
                    in_channels=input_channels,
                    num_classes=number_of_classes,
                    include_top=True,
                )

                model.train()

                logits = model(
                    input_tensor
                )

                self.assertEqual(
                    tuple(logits.shape),
                    (
                        batch_size,
                        number_of_classes,
                    ),
                )

                self.assertTrue(
                    torch.isfinite(logits).all()
                )

                loss = nn.CrossEntropyLoss()(
                    logits,
                    target_labels,
                )

                self.assertTrue(
                    torch.isfinite(loss)
                )

                model.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                trainable_parameters = [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]

                parameters_with_gradients = [
                    parameter
                    for parameter in trainable_parameters
                    if parameter.grad is not None
                ]

                self.assertTrue(
                    parameters_with_gradients,
                    (
                        f"{model_class.__name__} did not "
                        "produce parameter gradients."
                    ),
                )

                for parameter in parameters_with_gradients:
                    self.assertTrue(
                        torch.isfinite(
                            parameter.grad
                        ).all(),
                        (
                            f"{model_class.__name__} "
                            "produced non-finite gradients."
                        ),
                    )

                # This is how the framework switches a trained
                # classifier into an embedding extractor.
                model.include_top = False
                model.eval()

                with torch.no_grad():
                    embeddings = model(
                        input_tensor
                    )

                self.assertEqual(
                    embeddings.ndim,
                    2,
                )

                self.assertEqual(
                    embeddings.shape[0],
                    batch_size,
                )

                self.assertGreater(
                    embeddings.shape[1],
                    0,
                )

                self.assertTrue(
                    torch.isfinite(
                        embeddings
                    ).all()
                )

    def test_models_support_univariate_framework_input(self):
        """
        The default ECG pipeline produces tensors shaped (batch, 1, length).
        """
        input_tensor = torch.randn(
            3,
            1,
            64,
        )

        for model_class in PRODUCTION_MODELS:
            with self.subTest(
                model=model_class.__name__
            ):
                model = model_class(
                    in_channels=1,
                    num_classes=4,
                    include_top=True,
                )

                model.eval()

                with torch.no_grad():
                    logits = model(
                        input_tensor
                    )

                self.assertEqual(
                    tuple(logits.shape),
                    (3, 4),
                )

                self.assertTrue(
                    torch.isfinite(logits).all()
                )

    def test_each_model_completes_template_pipeline(self):
        """
        Exercise training, classifier-to-embedding switching, enrollment
        template creation, matching, and Rank-N evaluation for every model.
        """
        for model_class in PRODUCTION_MODELS:
            with self.subTest(
                model=model_class.__name__
            ):
                result = (
                    run.run_closed_set_identification(
                        self.synthetic_x,
                        self.synthetic_y,
                        model_class=model_class,
                        epochs=1,
                        batch_size=10,
                        lr=1e-3,
                        test_split=0.25,
                        val_split=0.0,
                        seed=42,
                        device="cpu",
                        visualize=False,
                        use_template=True,
                        template_fusion_method="mean",
                        template_size=2,
                        matching_method="cosine",
                        outlier_filtering_on_train=False,
                        outlier_filtering_on_test=False,
                        sqi_scores=None,
                        probe_fusion_size=1,
                        save_results_and_settings=False,
                        loader=None,
                        n_runs=1,
                        _return_stats=True,
                        intelligent_weight_loading=False,
                    )
                )

                metrics, data_statistics, hyperparameters = (
                    result
                )

                self.assertEqual(
                    len(metrics),
                    2,
                )

                rank_1 = float(
                    metrics[0]
                )

                self.assertTrue(
                    np.isfinite(rank_1)
                )
                self.assertGreaterEqual(
                    rank_1,
                    0.0,
                )
                self.assertLessEqual(
                    rank_1,
                    1.0,
                )

                # This compatibility fixture has exactly five
                # gallery identities, so the terminal Rank-5
                # point is defined in the CMC artifact but is
                # intentionally not exposed as a headline metric.
                self.assertIsNone(
                    metrics[1]
                )

                self.assertEqual(
                    data_statistics[
                        "Total Subjects"
                    ],
                    5,
                )

                self.assertGreater(
                    data_statistics[
                        "Train Samples"
                    ],
                    0,
                )

                self.assertGreater(
                    data_statistics[
                        "Test (Probe) Samples"
                    ],
                    0,
                )

                self.assertGreater(
                    hyperparameters[
                        "Total Model Parameters"
                    ],
                    0,
                )

                self.assertGreater(
                    hyperparameters[
                        "Trainable Model Parameters"
                    ],
                    0,
                )

                self.assertEqual(
                    hyperparameters[
                        "base_seed"
                    ],
                    42,
                )

                self.assertEqual(
                    hyperparameters[
                        "run_seeds"
                    ],
                    [42],
                )


if __name__ == "__main__":
    unittest.main()