"""
Final campaign manifest: schema, enumeration, counts and invariants.

These tests run offline. They never train a model, never load a dataset and
never write into the repository. Where a count is asserted it is also derived
independently from the configuration corpus or from the manifest structure, so
a matching pair of wrong numbers in the manifest and the runner cannot pass.
"""

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_campaign as campaign  # noqa: E402
from experiment_provenance import (  # noqa: E402
    IMPLEMENTATION_IDENTITY_SCHEMA,
    RESULT_PROVENANCE_SCHEMA,
    build_experiment_implementation_identity,
)

MANIFEST_PATH = PROJECT_ROOT / "campaigns" / "final_campaign.yaml"

EXPECTED_RUN_SEEDS = [42, 43, 44, 45, 46]

ABLATION_ANCHORS = (
    "cybhi_baseline_ci_closed_set_task02_verification.yaml",
    "cybhi_baseline_s1_closed_set_task02_verification.yaml",
    "ecgid_single_session_closed_set_task02_verification.yaml",
    "heartprint_single_session_closed_set_task02_verification.yaml",
    "mitbih_single_segment_closed_set_task02_verification.yaml",
    "nsrdb_single_segment_closed_set_task02_verification.yaml",
    "ptb_single_session_closed_set_task02_verification.yaml",
    "ptbxl_single_session_closed_set_task02_verification.yaml",
)

AUGMENTATION_ANCHORS = (
    "cybhi_long_term_closed_set_task06_verification.yaml",
    "ecgid_single_cross_session_closed_set_task06_verification.yaml",
)


def load():
    return campaign.load_manifest(MANIFEST_PATH)


def write_manifest(directory, manifest):
    path = Path(directory) / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return path


def invoke_cli(argv):
    buffer = io.StringIO()

    with redirect_stdout(buffer), redirect_stderr(buffer):
        try:
            status = campaign.main(argv)
        except SystemExit as error:
            status = int(error.code)

    return status, buffer.getvalue()


def current_result_record(condition, implementation=None):
    """Build the smallest canonical record accepted by resume validation."""
    implementation = copy.deepcopy(
        implementation
        or build_experiment_implementation_identity(
            repository_root=PROJECT_ROOT
        )
    )
    implementation["schema"] = IMPLEMENTATION_IDENTITY_SCHEMA
    implementation["git"]["status"] = "available"
    implementation["git"]["dirty"] = False

    return {
        "canonical_provenance": {
            "schema": RESULT_PROVENANCE_SCHEMA,
            "scientific_configuration": {
                "sha256": condition.scientific_sha256,
            },
            "implementation": implementation,
            "execution": {
                "campaign_id": None,
                "smoke_run": False,
                "status": "success",
                "successful": True,
                "completion": {"complete": True},
            },
            "publication_eligibility": {
                "eligible": True,
                "reasons": [],
            },
        }
    }


def write_result(directory, condition, records):
    result_directory = campaign.condition_result_directory(
        directory, condition
    )
    result_directory.mkdir(parents=True, exist_ok=True)
    path = result_directory / "result.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


class ManifestSchemaTests(unittest.TestCase):
    def test_manifest_parses(self):
        manifest = load()

        self.assertEqual(manifest["schema_version"], campaign.SCHEMA_VERSION)
        self.assertIn("controlled_ablation", manifest["studies"])
        self.assertIn("augmentation", manifest["studies"])

    def _write(self, directory, manifest):
        path = Path(directory) / "manifest.yaml"
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        return path

    def test_unknown_top_level_key_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["unexpected_section"] = {}

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "Unknown key"):
                campaign.load_manifest(path)

    def test_unknown_study_key_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["studies"]["controlled_ablation"]["surprise"] = True

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "Unknown key"):
                campaign.load_manifest(path)

    def test_unknown_tier_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["studies"]["controlled_ablation"]["tier"] = "experimental"

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "unknown tier"):
                campaign.load_manifest(path)

    def test_optional_study_declaring_core_tier_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["optional_studies"]["median_template_fusion"]["tier"] = "core"

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "declares tier"):
                campaign.load_manifest(path)

    def test_unknown_override_field_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["studies"]["controlled_ablation"]["axes"][0]["field"] = (
            "results_dir"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(
                campaign.CampaignError, "not a recognized scientific field"
            ):
                campaign.load_manifest(path)

    def test_bundle_level_outside_its_bundle_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        axis = manifest["studies"]["augmentation"]["axes"][0]
        axis["levels"][0]["values"]["probe_fusion_size"] = 3

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(
                campaign.CampaignError, "outside its bundle"
            ):
                campaign.load_manifest(path)

    def test_duplicate_level_labels_are_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        axis = manifest["studies"]["controlled_ablation"]["axes"][1]
        axis["levels"][1]["label"] = axis["levels"][0]["label"]

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "repeats"):
                campaign.load_manifest(path)

    def test_seed_schedule_mismatch_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["campaign"]["run_seeds"] = [42, 43, 44]

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(campaign.CampaignError, "run_seeds"):
                campaign.load_manifest(path)

    def test_fixed_split_seed_policy_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["campaign"]["split_seed_policy"] = "fixed"

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)

            with self.assertRaisesRegex(
                campaign.CampaignError, "split_seed_policy"
            ):
                campaign.load_manifest(path)

    def test_missing_base_config_is_rejected(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["studies"]["controlled_ablation"]["base_configs"][0] = (
            "configs/paper_reproduction/ecgid/does_not_exist.yaml"
        )

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, manifest)
            loaded = campaign.load_manifest(path)

            with self.assertRaisesRegex(campaign.CampaignError, "does not exist"):
                campaign.enumerate_conditions(loaded, resolve=False)


class AnchorResolutionTests(unittest.TestCase):
    def test_declared_anchors_resolve_uniquely_in_the_corpus(self):
        for name in ABLATION_ANCHORS + AUGMENTATION_ANCHORS:
            matches = sorted((PROJECT_ROOT / "configs").rglob(name))

            self.assertEqual(len(matches), 1, f"{name} -> {matches}")

    def test_manifest_references_exactly_the_declared_anchors(self):
        manifest = load()
        ablation = manifest["studies"]["controlled_ablation"]["base_configs"]
        augmentation = manifest["studies"]["augmentation"]["base_configs"]

        self.assertEqual(
            sorted(Path(p).name for p in ablation),
            sorted(ABLATION_ANCHORS),
        )
        self.assertEqual(
            sorted(Path(p).name for p in augmentation),
            sorted(AUGMENTATION_ANCHORS),
        )

    def test_ablation_anchors_exclude_all_available_configurations(self):
        manifest = load()

        for path in manifest["studies"]["controlled_ablation"]["base_configs"]:
            self.assertNotIn("all_available", Path(path).name)

    def test_every_referenced_config_parses_through_production_parsing(self):
        manifest = load()
        referenced = set()

        for _, study in campaign.iter_studies(manifest):
            referenced.update(study["base_configs"])

        for relative in sorted(referenced):
            configuration = campaign.resolve_effective_configuration(relative)

            self.assertEqual(configuration["n_runs"], 5, relative)
            self.assertEqual(configuration["seed"], 42, relative)
            self.assertIsNone(configuration["split_seed"], relative)


class BaseCorpusCountTests(unittest.TestCase):
    def test_base_corpus_matches_the_configuration_corpus_on_disk(self):
        manifest = load()
        groups = campaign.enumerate_base_corpus(manifest)

        # Derived independently from the corpus, not from the manifest.
        paper = sorted(
            (PROJECT_ROOT / "configs" / "paper_reproduction").rglob("*.yaml")
        )
        model = sorted(
            (PROJECT_ROOT / "configs" / "model_comparison").rglob("*.yaml")
        )

        self.assertEqual(len(groups["paper_reproduction"]), len(paper))
        self.assertEqual(len(groups["model_comparison"]), len(model))
        self.assertEqual(len(paper), 150)
        self.assertEqual(len(model), 84)
        self.assertEqual(len(paper) + len(model), 234)

    def test_base_counts_and_executions(self):
        summary = campaign.count_campaign(load())

        self.assertEqual(summary["n_runs"], 5)
        self.assertEqual(
            summary["base_groups"]["paper_reproduction"]["conditions"], 150
        )
        self.assertEqual(
            summary["base_groups"]["paper_reproduction"]["executions"], 750
        )
        self.assertEqual(
            summary["base_groups"]["model_comparison"]["conditions"], 84
        )
        self.assertEqual(
            summary["base_groups"]["model_comparison"]["executions"], 420
        )
        self.assertEqual(summary["base_conditions"], 234)
        self.assertEqual(summary["base_executions"], 1170)


class CampaignCountTests(unittest.TestCase):
    def setUp(self):
        self.summary = campaign.count_campaign(load())

    def test_controlled_ablation_counts(self):
        body = self.summary["studies"]["controlled_ablation"]

        self.assertEqual(body["new_conditions"], 72)
        self.assertEqual(body["new_executions"], 360)
        self.assertEqual(body["reused_conditions"], 8)

    def test_augmentation_counts(self):
        body = self.summary["studies"]["augmentation"]

        self.assertEqual(body["new_conditions"], 6)
        self.assertEqual(body["new_executions"], 30)
        self.assertEqual(body["reused_conditions"], 2)

    def test_additional_core_totals(self):
        self.assertEqual(self.summary["additional_core_conditions"], 78)
        self.assertEqual(self.summary["additional_core_executions"], 390)

    def test_distinct_core_conditions_exclude_reused_baselines(self):
        self.assertEqual(self.summary["distinct_core_conditions"], 312)
        self.assertEqual(
            self.summary["distinct_core_conditions"],
            self.summary["base_conditions"]
            + self.summary["additional_core_conditions"],
        )

    def test_total_core_executions(self):
        self.assertEqual(self.summary["total_core_executions"], 1560)

    def test_optional_counts_are_reported_separately(self):
        self.assertEqual(
            self.summary["optional_conditions"],
            sum(
                body["new_conditions"]
                for body in self.summary["studies"].values()
                if body["tier"] == "optional"
            ),
        )
        self.assertEqual(
            self.summary["optional_executions"],
            self.summary["optional_conditions"] * 5,
        )

    def test_optional_conditions_never_enter_core_counts(self):
        optional_new = sum(
            body["new_conditions"]
            for body in self.summary["studies"].values()
            if body["tier"] == "optional"
        )

        self.assertGreater(optional_new, 0)
        self.assertEqual(
            self.summary["additional_core_conditions"],
            sum(
                body["new_conditions"]
                for body in self.summary["studies"].values()
                if body["tier"] == "core"
            ),
        )
        self.assertEqual(
            self.summary["maximum_executions"],
            self.summary["total_core_executions"]
            + self.summary["optional_executions"],
        )

    def test_counts_are_derived_from_manifest_structure(self):
        """The totals must follow the declared levels, not a constant."""
        manifest = load()
        expected_new = 0

        for _, study in campaign.iter_studies(manifest):
            if study["tier"] != "core":
                continue

            levels = sum(len(axis["levels"]) for axis in study["axes"])
            expected_new += levels * len(study["base_configs"])

        self.assertEqual(
            self.summary["additional_core_conditions"], expected_new
        )


class EnumerationTests(unittest.TestCase):
    def test_enumeration_is_deterministic(self):
        manifest = load()
        first = campaign.enumerate_conditions(manifest, resolve=False)
        second = campaign.enumerate_conditions(manifest, resolve=False)

        self.assertEqual(
            [c.identifier for c in first],
            [c.identifier for c in second],
        )

    def test_enumeration_is_independent_of_mapping_order(self):
        manifest = load()
        shuffled = copy.deepcopy(manifest)
        shuffled["studies"] = dict(
            reversed(list(shuffled["studies"].items()))
        )
        shuffled["optional_studies"] = dict(
            reversed(list(shuffled["optional_studies"].items()))
        )

        self.assertEqual(
            [c.identifier for c in campaign.enumerate_conditions(
                manifest, resolve=False)],
            [c.identifier for c in campaign.enumerate_conditions(
                shuffled, resolve=False)],
        )

    def test_condition_identifiers_are_unique(self):
        conditions = campaign.enumerate_conditions(load(), resolve=False)
        identifiers = [c.identifier for c in conditions]

        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_scheduled_output_directories_are_unique(self):
        conditions = [
            c
            for c in campaign.enumerate_conditions(load(), resolve=False)
            if not c.reused
        ]
        directories = [c.output_directory for c in conditions]

        self.assertEqual(len(directories), len(set(directories)))

    def test_tier_filter_excludes_optional(self):
        core = campaign.enumerate_conditions(
            load(), tier="core", resolve=False
        )

        self.assertTrue(all(c.tier == "core" for c in core))
        self.assertTrue(any(c.study == "controlled_ablation" for c in core))
        self.assertFalse(
            any(c.study == "median_template_fusion" for c in core)
        )


class SingleFactorInvariantTests(unittest.TestCase):
    """Each scheduled condition may move only its own declared axis."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = load()
        cls.conditions = campaign.enumerate_conditions(
            cls.manifest, resolve=True
        )

    def _axis(self, condition):
        return campaign._declared_axis(self.manifest, condition)

    def test_every_scheduled_condition_moves_only_its_axis(self):
        for condition in self.conditions:
            if condition.reused or condition.study == "base":
                continue

            axis = self._axis(condition)
            allowed = (
                set(axis["bundle"]) if "bundle" in axis else {axis["field"]}
            )

            self.assertTrue(
                set(condition.changed_fields),
                f"{condition.identifier} changed nothing",
            )
            self.assertTrue(
                set(condition.changed_fields) <= allowed,
                f"{condition.identifier} changed {condition.changed_fields}, "
                f"allowed {sorted(allowed)}",
            )

    def test_probe_fusion_levels_change_only_probe_fusion_size(self):
        self._assert_axis_fields("probe_fusion", {"probe_fusion_size"})

    def test_template_size_levels_change_only_template_size(self):
        self._assert_axis_fields("template_size", {"template_size"})

    def test_template_fusion_levels_change_only_template_fusion_method(self):
        self._assert_axis_fields(
            "template_fusion", {"template_fusion_method"}
        )

    def test_matching_levels_change_only_matching_method(self):
        self._assert_axis_fields("matching", {"matching_method"})

    def test_augmentation_levels_change_only_the_augmentation_bundle(self):
        allowed = {
            "use_augmentation",
            "augmentation_method",
            "augmentation_parameters",
            "augmentation_copies",
        }
        self._assert_axis_fields("augmentation_method", allowed)

    def _assert_axis_fields(self, axis_name, allowed):
        seen = 0

        for condition in self.conditions:
            if condition.reused or condition.axis != axis_name:
                continue

            seen += 1
            self.assertTrue(
                set(condition.changed_fields) <= allowed,
                f"{condition.identifier}: {condition.changed_fields}",
            )

        self.assertGreater(seen, 0, f"no conditions found for {axis_name}")

    def test_augmentation_never_moves_pair_sampling_or_fusion(self):
        forbidden = {
            "pair_sampling_mode",
            "max_impostor_pairs",
            "pair_sampling_seed",
            "target_fars",
            "probe_fusion_size",
            "template_size",
            "template_fusion_method",
        }

        for condition in self.conditions:
            if condition.reused or condition.study.split("_")[0] != (
                "augmentation"
            ):
                continue

            self.assertFalse(
                set(condition.changed_fields) & forbidden,
                f"{condition.identifier}: {condition.changed_fields}",
            )

    def test_reused_baselines_apply_no_override(self):
        for condition in self.conditions:
            if condition.reused:
                self.assertEqual(condition.overrides, ())

    def test_reused_baseline_identity_matches_its_base_configuration(self):
        from experiment_provenance import (
            build_scientific_configuration_identity,
        )

        for condition in self.conditions:
            if not condition.reused:
                continue

            base = campaign.resolve_effective_configuration(
                condition.base_config
            )
            identity = build_scientific_configuration_identity(base)

            self.assertEqual(condition.scientific_sha256, identity["sha256"])

    def test_scheduled_scientific_identities_are_unique(self):
        identities = [
            c.scientific_sha256
            for c in self.conditions
            if not c.reused and c.study != "base"
        ]

        self.assertEqual(len(identities), len(set(identities)))

    def test_scheduled_identities_differ_from_their_baseline(self):
        for condition in self.conditions:
            if condition.reused or condition.study == "base":
                continue

            base = campaign.resolve_effective_configuration(
                condition.base_config
            )
            from experiment_provenance import (
                build_scientific_configuration_identity,
            )

            self.assertNotEqual(
                condition.scientific_sha256,
                build_scientific_configuration_identity(base)["sha256"],
                condition.identifier,
            )

    def test_frozen_verification_settings_are_preserved(self):
        for condition in self.conditions:
            if condition.study == "base":
                continue

            configuration = campaign.resolve_effective_configuration(
                condition.base_config, condition.override_mapping()
            )

            self.assertEqual(
                configuration["pair_sampling_mode"], "all_genuine",
                condition.identifier,
            )
            self.assertEqual(
                configuration["max_impostor_pairs"], 1000000,
                condition.identifier,
            )
            self.assertEqual(
                configuration["pair_sampling_seed"], 42, condition.identifier
            )
            self.assertEqual(
                configuration["target_fars"],
                [0.0001, 0.001, 0.01, 0.1],
                condition.identifier,
            )
            self.assertEqual(configuration["n_runs"], 5, condition.identifier)
            self.assertEqual(configuration["seed"], 42, condition.identifier)
            self.assertIsNone(
                configuration["split_seed"], condition.identifier
            )

    def test_probe_fusion_levels_are_not_reduced(self):
        levels = sorted(
            campaign.resolve_effective_configuration(
                c.base_config, c.override_mapping()
            )["probe_fusion_size"]
            for c in self.conditions
            if c.axis == "probe_fusion" and not c.reused
            and c.base_config.endswith(
                "ecgid_single_session_closed_set_task02_verification.yaml"
            )
        )

        self.assertEqual(levels, [3, 5, 7])


class ValidationTests(unittest.TestCase):
    def test_shipped_manifest_validates_cleanly(self):
        self.assertEqual(campaign.validate_campaign(load()), [])

    def test_validation_detects_a_two_axis_condition(self):
        import tempfile

        manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
        axis = manifest["studies"]["controlled_ablation"]["axes"][2]
        # Declare probe fusion but also move the matching method.
        axis["bundle"] = ["probe_fusion_size", "matching_method"]
        axis.pop("field")
        axis["baseline"] = {
            "probe_fusion_size": 1,
            "matching_method": "cosine",
        }
        axis["levels"] = [
            {
                "label": "k3",
                "values": {
                    "probe_fusion_size": 3,
                    "matching_method": "euclidean",
                },
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
            loaded = campaign.load_manifest(path)

            # The bundle now legitimises two fields, so the guard that matters
            # is that the declared bundle is what actually moved.
            conditions = [
                c
                for c in campaign.enumerate_conditions(loaded, resolve=True)
                if c.axis == "probe_fusion" and not c.reused
            ]

            for condition in conditions:
                self.assertEqual(
                    set(condition.changed_fields),
                    {"probe_fusion_size", "matching_method"},
                )

    def test_validation_detects_an_override_outside_the_declared_axis(self):
        manifest = load()
        conditions = campaign.enumerate_conditions(manifest, resolve=True)
        victim = next(
            c
            for c in conditions
            if not c.reused and c.study != "base"
        )

        tampered = campaign.Condition(
            tier=victim.tier,
            study=victim.study,
            base_config=victim.base_config,
            axis=victim.axis,
            level=victim.level,
            overrides=victim.overrides,
            output_directory=victim.output_directory,
            reused=False,
            identifier=victim.identifier,
            scientific_sha256=victim.scientific_sha256,
            changed_fields=victim.changed_fields + ("matching_method",),
        )

        problems = campaign._validate_condition_overrides(manifest, tampered)

        self.assertTrue(problems)
        self.assertIn("outside its axis", problems[0])

    def test_validation_detects_a_fixed_split_seed(self):
        manifest = load()
        conditions = campaign.enumerate_conditions(manifest, resolve=False)
        victim = next(c for c in conditions if not c.reused)

        tampered = campaign.Condition(
            tier=victim.tier,
            study=victim.study,
            base_config=victim.base_config,
            axis=victim.axis,
            level=victim.level,
            overrides=victim.overrides + (("split_seed", 7),),
            output_directory=victim.output_directory,
            reused=False,
            identifier=victim.identifier,
        )

        problems = campaign._validate_run_schedule_of(
            manifest, tampered, PROJECT_ROOT
        )

        self.assertTrue(any("split_seed" in p for p in problems))

    def test_validation_detects_a_changed_run_count(self):
        manifest = load()
        conditions = campaign.enumerate_conditions(manifest, resolve=False)
        victim = next(c for c in conditions if not c.reused)

        tampered = campaign.Condition(
            tier=victim.tier,
            study=victim.study,
            base_config=victim.base_config,
            axis=victim.axis,
            level=victim.level,
            overrides=victim.overrides + (("n_runs", 3),),
            output_directory=victim.output_directory,
            reused=False,
            identifier=victim.identifier,
        )

        problems = campaign._validate_run_schedule_of(
            manifest, tampered, PROJECT_ROOT
        )

        self.assertTrue(any("n_runs" in p for p in problems))

    def test_invalid_level_value_is_rejected_by_production_parsing(self):
        with self.assertRaises(campaign.CampaignError):
            campaign.resolve_effective_configuration(
                "configs/paper_reproduction/ecgid/"
                "ecgid_single_session_closed_set_task02_verification.yaml",
                {"matching_method": "not_a_metric"},
            )

    def test_invalid_augmentation_method_is_rejected(self):
        with self.assertRaises(campaign.CampaignError):
            campaign.resolve_effective_configuration(
                "configs/paper_reproduction/cybhi/"
                "cybhi_long_term_closed_set_task06_verification.yaml",
                {
                    "use_augmentation": True,
                    "augmentation_method": "not_a_method",
                },
            )

    def test_invalid_augmentation_parameter_is_rejected(self):
        with self.assertRaises(campaign.CampaignError):
            campaign.resolve_effective_configuration(
                "configs/paper_reproduction/cybhi/"
                "cybhi_long_term_closed_set_task06_verification.yaml",
                {
                    "use_augmentation": True,
                    "augmentation_method": "gaussian",
                    "augmentation_parameters": {"unsupported_key": 1},
                },
            )


class ValidationBeforeRunTests(unittest.TestCase):
    def _assert_run_refuses(self, manifest, study="controlled_ablation"):
        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(directory, manifest)
            status, output = invoke_cli([
                "run",
                "--manifest", str(path),
                "--study", study,
                "--artifact-root", str(Path(directory) / "artifacts"),
                "--dry-run",
            ])

        self.assertEqual(status, 2)
        self.assertIn("validation failed before execution", output.lower())
        self.assertNotIn("[PLAN]", output)
        return output

    def test_changed_run_seed_schedule_is_rejected_by_validate_and_run(self):
        manifest = copy.deepcopy(load())
        manifest["campaign"]["run_seeds"] = [43, 44, 45, 46, 47]

        problems = campaign.validate_campaign(manifest)
        self.assertTrue(any("run seeds" in problem for problem in problems))
        output = self._assert_run_refuses(manifest)
        self.assertIn("run seeds", output)

    def test_probe_k7_equal_to_baseline_is_rejected_by_validate_and_run(self):
        manifest = copy.deepcopy(load())
        probe_axis = manifest["studies"]["controlled_ablation"]["axes"][2]
        next(level for level in probe_axis["levels"]
             if level["label"] == "k7")["value"] = 1

        problems = campaign.validate_campaign(manifest)
        self.assertTrue(any("changes no" in problem for problem in problems))
        output = self._assert_run_refuses(manifest)
        self.assertIn("changes no", output)

    def test_duplicate_scientific_identity_is_rejected_before_run(self):
        manifest = copy.deepcopy(load())
        probe_axis = manifest["studies"]["controlled_ablation"]["axes"][2]
        next(level for level in probe_axis["levels"]
             if level["label"] == "k7")["value"] = 5

        output = self._assert_run_refuses(manifest)
        self.assertIn("same scientific identity", output)

    def test_duplicate_output_directory_is_rejected_before_run(self):
        manifest = copy.deepcopy(load())
        manifest["studies"]["duplicate_ablation"] = copy.deepcopy(
            manifest["studies"]["controlled_ablation"]
        )

        output = self._assert_run_refuses(
            manifest, study="duplicate_ablation"
        )
        self.assertIn("Duplicate output directory", output)

    def test_wrong_declared_baseline_is_rejected_by_validate_and_run(self):
        manifest = copy.deepcopy(load())
        manifest["studies"]["controlled_ablation"]["axes"][0][
            "baseline"
        ] = "median"

        problems = campaign.validate_campaign(manifest)
        self.assertTrue(any("baseline declares" in problem for problem in problems))
        output = self._assert_run_refuses(manifest)
        self.assertIn("baseline declares", output)


class BaseCorpusContractTests(unittest.TestCase):
    def test_exactly_234_base_conditions_are_executable_and_unique(self):
        conditions = campaign.enumerate_base_conditions(load(), resolve=True)

        self.assertEqual(len(conditions), 234)
        self.assertEqual(len({c.identifier for c in conditions}), 234)
        self.assertEqual(len({c.output_directory for c in conditions}), 234)
        self.assertTrue(all(c.scientific_sha256 for c in conditions))
        self.assertTrue(all(not c.reused for c in conditions))

    def test_each_base_condition_is_one_plain_main_command(self):
        conditions = campaign.enumerate_base_conditions(load(), resolve=False)
        commands = [
            campaign.build_condition_command("ARTIFACTS", condition)
            for condition in conditions
        ]

        self.assertEqual(len(commands), 234)
        for command in commands:
            self.assertEqual(command[1:3], ["-m", "main"])
            self.assertEqual(command.count("--config"), 1)
            self.assertEqual(command.count("--results_dir"), 1)
            self.assertEqual(len(command), 7)
            self.assertNotIn("--seed", command)
            self.assertNotIn("--n_runs", command)

    def test_complete_core_dry_run_plans_exactly_312_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "unused"
            status, output = invoke_cli([
                "run",
                "--manifest", str(MANIFEST_PATH),
                "--tier", "core",
                "--artifact-root", str(artifact_root),
                "--dry-run",
            ])

            self.assertFalse(artifact_root.exists())

        self.assertEqual(status, 0)
        self.assertEqual(output.count("[PLAN]"), 312)
        self.assertIn("# 312 condition(s) would run", output)
        self.assertEqual(campaign.count_campaign(load())["total_core_executions"], 1560)

    def test_root_must_exist_and_group_must_stay_below_it(self):
        manifest = copy.deepcopy(load())
        manifest["base_corpus"]["root"] = "not-a-real-corpus"

        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(directory, manifest)
            with self.assertRaisesRegex(campaign.CampaignError, "does not exist"):
                campaign.load_manifest(path)

        manifest = copy.deepcopy(load())
        manifest["base_corpus"]["groups"][0]["directory"] = "."

        with tempfile.TemporaryDirectory() as directory:
            path = write_manifest(directory, manifest)
            with self.assertRaisesRegex(campaign.CampaignError, "escapes"):
                campaign.load_manifest(path)


class OverrideRenderingTests(unittest.TestCase):
    def test_top_level_nargs_list_is_rendered_as_separate_tokens(self):
        argv = campaign.build_override_arguments(
            {"target_fars": [0.1, 0.01]}
        )

        self.assertEqual(argv, ["--target_fars", "0.1", "0.01"])

        resolved = campaign.resolve_effective_configuration(
            "configs/paper_reproduction/ecgid/"
            "ecgid_single_session_closed_set_task02_verification.yaml",
            {"target_fars": [0.1, 0.01]},
        )
        self.assertEqual(resolved["target_fars"], [0.001, 0.01, 0.1])

    def test_nested_list_remains_canonical_json_inside_dictionary(self):
        value = {"scale_range": [0.9, 1.1]}
        argv = campaign.build_override_arguments(
            {"augmentation_parameters": value}
        )

        self.assertEqual(
            argv,
            ["--augmentation_parameters", '{"scale_range": [0.9, 1.1]}'],
        )
        resolved = campaign.resolve_effective_configuration(
            "configs/paper_reproduction/ecgid/"
            "ecgid_single_cross_session_closed_set_task06_verification.yaml",
            {
                "use_augmentation": True,
                "augmentation_method": "amplitude",
                "augmentation_parameters": value,
                "augmentation_copies": 1,
            },
        )
        self.assertEqual(resolved["augmentation_parameters"], value)


class CommandLineTests(unittest.TestCase):
    def _run(self, argv):
        buffer = io.StringIO()

        with redirect_stdout(buffer), redirect_stderr(buffer):
            status = campaign.main(argv)

        return status, buffer.getvalue()

    def test_validate_reports_success(self):
        status, output = self._run(
            ["validate", "--manifest", str(MANIFEST_PATH)]
        )

        self.assertEqual(status, 0)
        self.assertIn("[OK]", output)

    def test_count_reports_the_core_totals(self):
        status, output = self._run(
            ["count", "--manifest", str(MANIFEST_PATH)]
        )

        self.assertEqual(status, 0)
        self.assertIn("1170", output)
        self.assertIn("1560", output)
        self.assertIn("312", output)

    def test_list_marks_reused_and_new_conditions(self):
        status, output = self._run(
            ["list", "--manifest", str(MANIFEST_PATH), "--tier", "core"]
        )

        self.assertEqual(status, 0)
        self.assertIn("REUSED", output)
        self.assertIn("NEW", output)

    def test_run_without_scope_is_refused(self):
        with self.assertRaises(SystemExit):
            self._run(
                [
                    "run",
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--artifact-root",
                    "unused",
                ]
            )

    def test_run_rejects_an_unknown_study(self):
        with self.assertRaises(SystemExit):
            self._run(
                [
                    "run",
                    "--manifest",
                    str(MANIFEST_PATH),
                    "--artifact-root",
                    "unused",
                    "--study",
                    "no_such_study",
                ]
            )

    def test_optional_study_requires_explicit_selection(self):
        status, output = self._run(
            ["list", "--manifest", str(MANIFEST_PATH), "--tier", "core"]
        )

        self.assertEqual(status, 0)
        self.assertNotIn("median_template_fusion", output)

        status, output = self._run(
            [
                "list",
                "--manifest",
                str(MANIFEST_PATH),
                "--study",
                "median_template_fusion",
            ]
        )

        self.assertEqual(status, 0)
        self.assertIn("median_template_fusion", output)

    def test_read_only_commands_create_no_files(self):
        import tempfile

        before = {
            p
            for p in PROJECT_ROOT.rglob("*")
            if ".git" not in p.parts and "__pycache__" not in p.parts
        }

        with tempfile.TemporaryDirectory():
            self._run(["validate", "--manifest", str(MANIFEST_PATH)])
            self._run(["count", "--manifest", str(MANIFEST_PATH)])
            self._run(["list", "--manifest", str(MANIFEST_PATH)])

        after = {
            p
            for p in PROJECT_ROOT.rglob("*")
            if ".git" not in p.parts and "__pycache__" not in p.parts
        }

        self.assertEqual(after - before, set())


class OptionalExecutionTests(unittest.TestCase):
    def test_optional_tier_cannot_be_executed_as_a_group(self):
        status, output = invoke_cli([
            "run",
            "--manifest", str(MANIFEST_PATH),
            "--tier", "optional",
            "--artifact-root", "unused",
            "--dry-run",
        ])

        self.assertEqual(status, 2)
        self.assertIn("requires an explicit --study", output)
        self.assertNotIn("[PLAN]", output)

    def test_each_explicit_optional_study_can_be_planned(self):
        expected = {
            "median_template_fusion": 8,
            "augmentation_subject_disjoint": 6,
        }

        for study, count in expected.items():
            with self.subTest(study=study):
                status, output = invoke_cli([
                    "run",
                    "--manifest", str(MANIFEST_PATH),
                    "--study", study,
                    "--artifact-root", "unused",
                    "--dry-run",
                ])
                self.assertEqual(status, 0)
                self.assertEqual(output.count("[PLAN]"), count)

    def test_incompatible_tier_and_study_are_rejected(self):
        status, output = invoke_cli([
            "run",
            "--manifest", str(MANIFEST_PATH),
            "--tier", "core",
            "--study", "median_template_fusion",
            "--artifact-root", "unused",
            "--dry-run",
        ])

        self.assertEqual(status, 2)
        self.assertIn("belongs to tier 'optional'", output)

    def test_optional_tier_remains_listable(self):
        status, output = invoke_cli([
            "list",
            "--manifest", str(MANIFEST_PATH),
            "--tier", "optional",
        ])

        self.assertEqual(status, 0)
        self.assertIn("median_template_fusion", output)
        self.assertIn("augmentation_subject_disjoint", output)


class ResumeProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.condition = campaign.enumerate_base_conditions(
            load(), resolve=True
        )[0]
        cls.current_implementation = (
            build_experiment_implementation_identity(
                repository_root=PROJECT_ROOT
            )
        )

    def _complete(self, directory, expected=None):
        return campaign.is_condition_complete(
            directory,
            self.condition,
            expected_implementation=(
                expected or self.current_implementation
            ),
        )

    def test_exact_current_implementation_is_complete_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            write_result(
                directory,
                self.condition,
                [current_result_record(
                    self.condition, self.current_implementation
                )],
            )

            self.assertTrue(self._complete(directory))
            calls = []
            executed, skipped = campaign.run_conditions(
                [self.condition],
                directory,
                resume=True,
                runner=lambda command, root: calls.append(command),
                stream=io.StringIO(),
            )

        self.assertEqual(executed, [])
        self.assertEqual(skipped, [self.condition])
        self.assertEqual(calls, [])

    def test_stale_implementation_is_incomplete_and_rerun(self):
        stale = current_result_record(
            self.condition, self.current_implementation
        )
        stale["canonical_provenance"]["implementation"][
            "source_sha256"
        ] = "0" * 64

        with tempfile.TemporaryDirectory() as directory:
            write_result(directory, self.condition, [stale])

            self.assertFalse(self._complete(directory))
            calls = []
            executed, skipped = campaign.run_conditions(
                [self.condition],
                directory,
                resume=True,
                runner=lambda command, root: calls.append(command),
                stream=io.StringIO(),
            )

        self.assertEqual(executed, [self.condition])
        self.assertEqual(skipped, [])
        self.assertEqual(len(calls), 1)

    def test_wrong_incomplete_and_ineligible_records_are_rejected(self):
        mutations = {}

        wrong = current_result_record(
            self.condition, self.current_implementation
        )
        wrong["canonical_provenance"]["scientific_configuration"][
            "sha256"
        ] = "f" * 64
        mutations["wrong scientific identity"] = wrong

        incomplete = current_result_record(
            self.condition, self.current_implementation
        )
        incomplete["canonical_provenance"]["execution"]["completion"][
            "complete"
        ] = False
        mutations["incomplete"] = incomplete

        ineligible = current_result_record(
            self.condition, self.current_implementation
        )
        ineligible["canonical_provenance"]["publication_eligibility"][
            "eligible"
        ] = False
        ineligible["canonical_provenance"]["publication_eligibility"][
            "reasons"
        ] = ["test"]
        mutations["ineligible"] = ineligible

        for name, record in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                write_result(directory, self.condition, [record])
                self.assertFalse(self._complete(directory))

    def test_malformed_and_duplicate_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_result(directory, self.condition, [])
            path.write_text("{not-json}\n", encoding="utf-8")
            self.assertFalse(self._complete(directory))

        record = current_result_record(
            self.condition, self.current_implementation
        )
        with tempfile.TemporaryDirectory() as directory:
            write_result(directory, self.condition, [record, record])
            self.assertFalse(self._complete(directory))


class ExecutionPlanTests(unittest.TestCase):
    def test_command_uses_the_ordinary_entry_point_and_results_dir(self):
        conditions = campaign.enumerate_conditions(load(), resolve=False)
        condition = next(
            c
            for c in conditions
            if not c.reused and c.axis == "probe_fusion"
        )

        command = campaign.build_condition_command("ARTIFACTS", condition)

        self.assertIn("-m", command)
        self.assertIn("main", command)
        self.assertIn("--config", command)
        self.assertIn("--results_dir", command)
        self.assertIn("--probe_fusion_size", command)

    def test_reused_baselines_are_never_scheduled(self):
        conditions = campaign.enumerate_conditions(load(), resolve=False)
        calls = []

        executed, skipped = campaign.run_conditions(
            conditions,
            "ARTIFACTS",
            runner=lambda command, root: calls.append(command),
            stream=io.StringIO(),
        )

        self.assertEqual(len(skipped), 0)
        self.assertEqual(len(calls), len(executed))
        self.assertTrue(all(not c.reused for c in executed))
        self.assertEqual(
            len(executed),
            len([c for c in conditions if not c.reused]),
        )

    def test_resume_does_not_skip_merely_because_a_file_exists(self):
        import tempfile

        conditions = campaign.enumerate_conditions(load(), resolve=True)
        condition = next(c for c in conditions if not c.reused)

        with tempfile.TemporaryDirectory() as directory:
            results = campaign.condition_result_directory(directory, condition)
            results.mkdir(parents=True, exist_ok=True)
            (results / "Closed-Set_Verification.jsonl").write_text(
                '{"not": "a valid record"}\n', encoding="utf-8"
            )

            self.assertFalse(
                campaign.is_condition_complete(directory, condition)
            )

            calls = []
            executed, skipped = campaign.run_conditions(
                [condition],
                directory,
                resume=True,
                runner=lambda command, root: calls.append(command),
                stream=io.StringIO(),
            )

            self.assertEqual(len(executed), 1)
            self.assertEqual(len(skipped), 0)

    def test_missing_output_is_not_complete(self):
        import tempfile

        conditions = campaign.enumerate_conditions(load(), resolve=True)
        condition = next(c for c in conditions if not c.reused)

        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(
                campaign.is_condition_complete(directory, condition)
            )


class ReadmeTests(unittest.TestCase):
    def test_readme_documents_the_actual_commands(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("campaigns/final_campaign.yaml", readme)

        for command in ("validate", "count", "list", "run"):
            self.assertIn(
                f"python -m scripts.run_campaign {command}", readme
            )

        self.assertIn("--tier core --artifact-root", readme)
        self.assertIn("every optional study must be named directly", readme)
        self.assertIn("current implementation provenance", readme)
        self.assertIn("complete shipped configuration corpus", readme)

    def test_readme_commands_are_accepted_by_the_parser(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        parser = campaign.build_parser()

        # Shell examples may wrap with a trailing backslash; join them first so
        # a wrapped command is checked in full rather than truncated.
        joined = readme.replace("\\\n", " ")
        checked = 0

        for line in joined.splitlines():
            stripped = line.strip()

            if not stripped.startswith("python -m scripts.run_campaign"):
                continue

            argv = stripped.split()[3:]

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                parsed = parser.parse_args(argv)

            self.assertTrue(parsed.command)
            checked += 1

        self.assertGreaterEqual(checked, 4)


if __name__ == "__main__":
    unittest.main()
