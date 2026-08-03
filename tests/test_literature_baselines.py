import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main
import models

# Beat length produced by each dataset at pre_s=0.2, post_s=0.4.
DATASET_BEAT_LENGTHS = {
    "nsrdb": 76,
    "heartprint": 150,
    "mitbih": 216,
    "ecgid": 300,
    "ptbxl": 300,
    "ptb": 600,
}

LITERATURE_BASELINES = {
    "ecgxtractor": models.ECGXtractor,
    "mobilenet_gru": models.MobileNetGRU,
    "multiscale_cnn": models.MultiScaleCNN,
    "separable_resnet": models.SeparableResNet,
}


class RegistryTests(unittest.TestCase):
    """
    Every selectable architecture must be reachable and constructible.
    """

    def test_registry_backs_the_cli_choices(self):
        parser = main.get_parser()

        model_action = next(
            action
            for action in parser._actions
            if action.dest == "model"
        )

        self.assertEqual(
            set(model_action.choices),
            set(main.MODEL_REGISTRY),
        )

    def test_default_model_is_unchanged(self):
        parser = main.get_parser()

        model_action = next(
            action
            for action in parser._actions
            if action.dest == "model"
        )

        self.assertEqual(model_action.default, "deepecg")

    def test_literature_baselines_are_registered(self):
        for name, cls in LITERATURE_BASELINES.items():
            with self.subTest(model=name):
                self.assertIn(name, main.MODEL_REGISTRY)
                self.assertIs(
                    main.MODEL_REGISTRY[name],
                    cls,
                )

    def test_previously_available_models_are_retained(self):
        for name in (
            "deepecg",
            "resnet1d",
            "rnn",
            "hybrid",
            "transformer",
        ):
            with self.subTest(model=name):
                self.assertIn(name, main.MODEL_REGISTRY)


class ModelContractTests(unittest.TestCase):
    """
    Every architecture must satisfy the interface the runners rely on.
    """

    def build(self, cls, include_top, num_classes=7, in_channels=1):
        model = cls(
            in_channels=in_channels,
            num_classes=num_classes,
            include_top=include_top,
        )
        model.eval()

        return model

    def test_classification_head_returns_logits(self):
        for name, cls in main.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                model = self.build(cls, include_top=True)

                with torch.no_grad():
                    logits = model(torch.randn(4, 1, 216))

                self.assertEqual(logits.shape, (4, 7))

    def test_headless_model_returns_two_dimensional_embeddings(self):
        for name, cls in main.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                model = self.build(cls, include_top=False)

                with torch.no_grad():
                    embedding = model(torch.randn(4, 1, 216))

                self.assertEqual(embedding.ndim, 2)
                self.assertEqual(embedding.shape[0], 4)
                self.assertGreater(embedding.shape[1], 0)

    def test_every_dataset_beat_length_is_accepted(self):
        for name, cls in main.MODEL_REGISTRY.items():
            for dataset, length in DATASET_BEAT_LENGTHS.items():
                with self.subTest(model=name, dataset=dataset):
                    model = self.build(cls, include_top=True)

                    with torch.no_grad():
                        logits = model(
                            torch.randn(2, 1, length)
                        )

                    self.assertEqual(logits.shape, (2, 7))

    def test_embedding_dimension_is_length_independent(self):
        # A template built on one dataset must be comparable in shape to one
        # built on another, so pooling must absorb the length difference.
        for name, cls in main.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                dimensions = set()

                for length in DATASET_BEAT_LENGTHS.values():
                    model = self.build(cls, include_top=False)

                    with torch.no_grad():
                        dimensions.add(
                            model(
                                torch.randn(2, 1, length)
                            ).shape[1]
                        )

                self.assertEqual(len(dimensions), 1)

    def test_outputs_are_finite(self):
        for name, cls in main.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                model = self.build(cls, include_top=False)

                with torch.no_grad():
                    embedding = model(torch.randn(4, 1, 216))

                self.assertTrue(
                    torch.isfinite(embedding).all()
                )

    def test_multichannel_input_is_accepted(self):
        # MIT-BIH and NSRDB can supply two leads.
        for name, cls in main.MODEL_REGISTRY.items():
            with self.subTest(model=name):
                model = self.build(
                    cls,
                    include_top=True,
                    in_channels=2,
                )

                with torch.no_grad():
                    logits = model(torch.randn(2, 2, 216))

                self.assertEqual(logits.shape, (2, 7))

    def test_gradients_reach_every_parameter(self):
        for name, cls in LITERATURE_BASELINES.items():
            with self.subTest(model=name):
                model = cls(
                    in_channels=1,
                    num_classes=7,
                    include_top=True,
                )
                model.train()

                logits = model(torch.randn(4, 1, 216))
                logits.sum().backward()

                unused = [
                    parameter_name
                    for parameter_name, parameter in (
                        model.named_parameters()
                    )
                    if parameter.grad is None
                ]

                self.assertEqual(unused, [])


class ECGXtractorReconstructionTests(unittest.TestCase):
    """
    The autoencoder branch must reconstruct the input shape.
    """

    def test_decode_restores_the_input_shape(self):
        model = models.ECGXtractor(
            in_channels=1,
            num_classes=7,
            include_decoder=True,
        )
        model.eval()

        # Three stride-2 pooling stages and three stride-2 transposes, so a
        # length divisible by 8 round-trips exactly.
        signal = torch.randn(2, 1, 216)

        with torch.no_grad():
            reconstruction = model.decode(signal)

        self.assertEqual(
            reconstruction.shape,
            signal.shape,
        )

    def test_decoder_is_absent_by_default(self):
        # Supervised training never uses it, so it must not add parameters
        # to the reported computational profile.
        model = models.ECGXtractor(
            in_channels=1,
            num_classes=7,
        )

        self.assertIsNone(model.decoder)

        with self.assertRaises(RuntimeError) as raised:
            model.decode(torch.randn(2, 1, 216))

        self.assertIn(
            "include_decoder=True",
            str(raised.exception),
        )

    def test_decoder_adds_parameters_only_when_requested(self):
        without = models.ECGXtractor(num_classes=7)
        with_decoder = models.ECGXtractor(
            num_classes=7,
            include_decoder=True,
        )

        self.assertLess(
            sum(p.numel() for p in without.parameters()),
            sum(
                p.numel()
                for p in with_decoder.parameters()
            ),
        )

    def test_encode_matches_headless_forward(self):
        model = models.ECGXtractor(
            in_channels=1,
            num_classes=7,
            include_top=False,
        )
        model.eval()

        signal = torch.randn(2, 1, 216)

        with torch.no_grad():
            self.assertTrue(
                torch.allclose(
                    model.encode(signal),
                    model(signal),
                )
            )


class ParameterBudgetTests(unittest.TestCase):
    """
    The lightweight baselines must actually be lightweight.
    """

    def parameter_count(self, cls):
        model = cls(in_channels=1, num_classes=100)

        return sum(
            parameter.numel()
            for parameter in model.parameters()
        )

    def test_separable_models_are_smaller_than_resnet1d(self):
        resnet_parameters = self.parameter_count(
            models.ResNet1D
        )

        for cls in (
            models.MobileNetGRU,
            models.SeparableResNet,
        ):
            with self.subTest(model=cls.__name__):
                self.assertLess(
                    self.parameter_count(cls),
                    resnet_parameters,
                )

    def test_width_multiplier_scales_mobilenet(self):
        narrow = models.MobileNetGRU(
            width_multiplier=0.5,
        )
        wide = models.MobileNetGRU(
            width_multiplier=1.0,
        )

        self.assertLess(
            sum(p.numel() for p in narrow.parameters()),
            sum(p.numel() for p in wide.parameters()),
        )


class MultiScaleBranchTests(unittest.TestCase):
    """
    The multi-scale model must genuinely use several receptive fields.
    """

    def test_default_uses_three_distinct_kernel_sizes(self):
        model = models.MultiScaleCNN()

        self.assertEqual(len(model.branches), 3)

    def test_kernel_sizes_are_configurable(self):
        model = models.MultiScaleCNN(
            kernel_sizes=(5, 11),
        )

        self.assertEqual(len(model.branches), 2)

        with torch.no_grad():
            output = model(torch.randn(2, 1, 216))

        self.assertEqual(output.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
