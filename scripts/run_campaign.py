"""
Enumerate and execute the declared experiment campaign.

The campaign manifest describes the shipped configuration corpus and explicit
study overrides. This module resolves that description into a deterministic
list of conditions, validates it without running anything, and executes a
selected part of it by invoking the ordinary experiment entry point once per
condition.

It contains no evaluation logic. Every scientific value comes from the
referenced configuration or from the manifest, and each condition is executed
exactly as if its configuration had been written out by hand, so the resulting
records carry the same configuration identity and provenance as any other
experiment.

Usage:

    python -m scripts.run_campaign validate --manifest campaigns/final_campaign.yaml
    python -m scripts.run_campaign count --manifest campaigns/final_campaign.yaml
    python -m scripts.run_campaign list --manifest campaigns/final_campaign.yaml --tier core
    python -m scripts.run_campaign run --manifest campaigns/final_campaign.yaml --study controlled_ablation
"""

import argparse
import io
import json
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as experiment_main  # noqa: E402
from experiment_provenance import (  # noqa: E402
    EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS,
    ResultCollectionError,
    build_experiment_implementation_identity,
    build_scientific_configuration_identity,
    classify_effective_configuration_fields,
    select_exact_result_record,
)

SCHEMA_VERSION = 1

TIERS = ("core", "optional")

MANIFEST_KEYS = {
    "schema_version",
    "campaign",
    "base_corpus",
    "studies",
    "optional_studies",
}

CAMPAIGN_KEYS = {
    "name",
    "description",
    "n_runs",
    "run_seeds",
    "split_seed_policy",
}

BASE_CORPUS_KEYS = {"tier", "root", "groups"}
BASE_GROUP_KEYS = {"name", "directory", "expected_conditions"}

STUDY_KEYS = {
    "tier",
    "description",
    "output_namespace",
    "base_configs",
    "axes",
}

SINGLE_FIELD_AXIS_KEYS = {"name", "field", "baseline", "levels"}
BUNDLE_AXIS_KEYS = {"name", "bundle", "baseline", "levels"}

SINGLE_FIELD_LEVEL_KEYS = {"label", "value"}
BUNDLE_LEVEL_KEYS = {"label", "values"}

# Overrides may only touch fields the framework already treats as scientific.
# Anything else would move a condition without moving its identity.
OVERRIDABLE_FIELDS = frozenset(EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS)


class CampaignError(Exception):
    """Raised when the manifest is malformed or cannot be resolved."""


@dataclass(frozen=True)
class Condition:
    """One scientific configuration scheduled for execution."""

    tier: str
    study: str
    base_config: str
    axis: str
    level: str
    overrides: tuple
    output_directory: str
    reused: bool
    identifier: str
    scientific_sha256: str = ""
    changed_fields: tuple = field(default=())

    def override_mapping(self):
        return dict(self.overrides)


# ---------------------------------------------------------------------------
# Manifest loading and schema validation
# ---------------------------------------------------------------------------
def load_manifest(path, project_root=PROJECT_ROOT):
    """Read and schema-check the campaign manifest."""
    path = Path(path)

    if not path.is_file():
        raise CampaignError(f"Campaign manifest does not exist: {path}")

    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(manifest, dict):
        raise CampaignError("Campaign manifest must be a mapping.")

    _reject_unknown(manifest, MANIFEST_KEYS, "manifest")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CampaignError(
            f"Unsupported campaign schema_version: "
            f"{manifest.get('schema_version')!r}. Expected {SCHEMA_VERSION}."
        )

    for required in ("campaign", "base_corpus", "studies"):
        if required not in manifest:
            raise CampaignError(f"Campaign manifest is missing '{required}'.")

    campaign = manifest["campaign"]
    if not isinstance(campaign, dict):
        raise CampaignError("'campaign' must be a mapping.")
    _reject_unknown(campaign, CAMPAIGN_KEYS, "campaign")

    _validate_run_schedule(campaign)

    base_corpus = manifest["base_corpus"]
    if not isinstance(base_corpus, dict):
        raise CampaignError("'base_corpus' must be a mapping.")
    _reject_unknown(base_corpus, BASE_CORPUS_KEYS, "base_corpus")

    if base_corpus.get("tier") != "core":
        raise CampaignError("'base_corpus' must declare tier 'core'.")

    corpus_root = base_corpus.get("root")
    if not isinstance(corpus_root, str) or not corpus_root.strip():
        raise CampaignError("base_corpus.root must be a non-empty path.")
    if Path(corpus_root).is_absolute():
        raise CampaignError("base_corpus.root must be relative to the project.")

    resolved_project_root = project_root.resolve()
    resolved_root = (project_root / corpus_root).resolve()

    try:
        resolved_root.relative_to(resolved_project_root)
    except ValueError as error:
        raise CampaignError(
            f"base_corpus.root escapes the project: {corpus_root}"
        ) from error

    if not resolved_root.is_dir():
        raise CampaignError(
            f"base_corpus.root does not exist: {corpus_root}"
        )

    for group in base_corpus.get("groups") or []:
        if not isinstance(group, dict):
            raise CampaignError("Each base corpus group must be a mapping.")
        _reject_unknown(group, BASE_GROUP_KEYS, "base_corpus group")

        if "directory" in group:
            try:
                resolved_dir = (project_root / group["directory"]).resolve()
                resolved_dir.relative_to(resolved_root)
            except ValueError as error:
                raise CampaignError(
                    f"Base corpus group '{group.get('name')}' directory "
                    f"'{group['directory']}' escapes base_corpus.root "
                    f"'{corpus_root}'."
                ) from error

    for section, expected_tier in (
        ("studies", "core"),
        ("optional_studies", "optional"),
    ):
        studies = manifest.get(section) or {}

        if not isinstance(studies, dict):
            raise CampaignError(f"'{section}' must be a mapping.")

        for name, study in studies.items():
            _validate_study(name, study, expected_tier)

    _reject_duplicate_study_names(manifest)

    return manifest


def _reject_unknown(mapping, allowed, where):
    unknown = sorted(set(mapping) - set(allowed))

    if unknown:
        raise CampaignError(
            f"Unknown key(s) in {where}: {', '.join(unknown)}."
        )


def _validate_run_schedule(campaign):
    run_count = campaign.get("n_runs")
    seeds = campaign.get("run_seeds")

    if not isinstance(run_count, int) or isinstance(run_count, bool):
        raise CampaignError("campaign.n_runs must be an integer.")

    if not isinstance(seeds, list) or not seeds:
        raise CampaignError("campaign.run_seeds must be a non-empty list.")

    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise CampaignError("campaign.run_seeds must contain integers.")

    if len(seeds) != run_count:
        raise CampaignError(
            f"campaign.run_seeds lists {len(seeds)} seeds but n_runs is "
            f"{run_count}."
        )

    if campaign.get("split_seed_policy") != "inherit":
        raise CampaignError(
            "campaign.split_seed_policy must be 'inherit'; the shipped "
            "configurations leave the split seed following each run seed."
        )


def _validate_study(name, study, expected_tier):
    if not isinstance(study, dict):
        raise CampaignError(f"Study '{name}' must be a mapping.")

    _reject_unknown(study, STUDY_KEYS, f"study '{name}'")

    tier = study.get("tier")

    if tier not in TIERS:
        raise CampaignError(f"Study '{name}' has unknown tier {tier!r}.")

    if tier != expected_tier:
        raise CampaignError(
            f"Study '{name}' declares tier {tier!r} but is defined in the "
            f"{expected_tier} section."
        )

    if not study.get("output_namespace"):
        raise CampaignError(f"Study '{name}' is missing 'output_namespace'.")

    if not study.get("base_configs"):
        raise CampaignError(f"Study '{name}' declares no base configs.")

    axes = study.get("axes")

    if not isinstance(axes, list) or not axes:
        raise CampaignError(f"Study '{name}' declares no axes.")

    for axis in axes:
        _validate_axis(name, axis)


def _validate_axis(study_name, axis):
    if not isinstance(axis, dict):
        raise CampaignError(f"Study '{study_name}' has a non-mapping axis.")

    is_bundle = "bundle" in axis

    if "baseline" not in axis:
        raise CampaignError(
            f"Axis '{axis.get('name')}' in study '{study_name}' is missing "
            "its baseline declaration."
        )

    if is_bundle:
        _reject_unknown(axis, BUNDLE_AXIS_KEYS, f"axis in study '{study_name}'")
        fields = axis.get("bundle")

        if not isinstance(fields, list) or not fields:
            raise CampaignError(
                f"Axis '{axis.get('name')}' declares an empty bundle."
            )

        baseline = axis["baseline"]
        if not isinstance(baseline, dict) or not baseline:
            raise CampaignError(
                f"Axis '{axis.get('name')}' declares an invalid bundle "
                "baseline."
            )

        unknown_baseline = sorted(set(baseline) - set(fields))
        if unknown_baseline:
            raise CampaignError(
                f"Axis '{axis.get('name')}' baseline sets field(s) outside "
                f"its bundle: {', '.join(unknown_baseline)}."
            )
    else:
        _reject_unknown(
            axis, SINGLE_FIELD_AXIS_KEYS, f"axis in study '{study_name}'"
        )
        fields = [axis.get("field")]

        if not axis.get("field"):
            raise CampaignError(
                f"Axis '{axis.get('name')}' in study '{study_name}' declares "
                "neither 'field' nor 'bundle'."
            )

    for name in fields:
        if name not in OVERRIDABLE_FIELDS:
            raise CampaignError(
                f"Axis '{axis.get('name')}' in study '{study_name}' overrides "
                f"{name!r}, which is not a recognized scientific field."
            )

    levels = axis.get("levels")

    if not isinstance(levels, list) or not levels:
        raise CampaignError(
            f"Axis '{axis.get('name')}' in study '{study_name}' has no levels."
        )

    labels = []

    for level in levels:
        if not isinstance(level, dict):
            raise CampaignError(
                f"Axis '{axis.get('name')}' has a non-mapping level."
            )

        allowed = BUNDLE_LEVEL_KEYS if is_bundle else SINGLE_FIELD_LEVEL_KEYS
        _reject_unknown(level, allowed, f"level in axis '{axis.get('name')}'")

        label = level.get("label")

        if not label:
            raise CampaignError(
                f"A level of axis '{axis.get('name')}' has no label."
            )

        labels.append(label)

        if is_bundle:
            values = level.get("values")

            if not isinstance(values, dict) or not values:
                raise CampaignError(
                    f"Level '{label}' of axis '{axis.get('name')}' declares no "
                    "values."
                )

            unknown = sorted(set(values) - set(fields))

            if unknown:
                raise CampaignError(
                    f"Level '{label}' of axis '{axis.get('name')}' sets "
                    f"field(s) outside its bundle: {', '.join(unknown)}."
                )
        elif "value" not in level:
            raise CampaignError(
                f"Level '{label}' of axis '{axis.get('name')}' has no value."
            )

    duplicates = sorted({x for x in labels if labels.count(x) > 1})

    if duplicates:
        raise CampaignError(
            f"Axis '{axis.get('name')}' in study '{study_name}' repeats "
            f"level label(s): {', '.join(duplicates)}."
        )


def _reject_duplicate_study_names(manifest):
    core = set(manifest.get("studies") or {})
    optional = set(manifest.get("optional_studies") or {})
    shared = sorted(core & optional)

    if shared:
        raise CampaignError(
            f"Study name(s) declared in both tiers: {', '.join(shared)}."
        )


def iter_studies(manifest):
    """Yield ``(name, study)`` for every declared study, core tier first."""
    for section in ("studies", "optional_studies"):
        for name in sorted(manifest.get(section) or {}):
            yield name, manifest[section][name]


# ---------------------------------------------------------------------------
# Base corpus
# ---------------------------------------------------------------------------
def enumerate_base_corpus(manifest, project_root=PROJECT_ROOT):
    """Return the shipped configuration paths, grouped and sorted."""
    groups = {}

    for group in manifest["base_corpus"].get("groups") or []:
        directory = project_root / group["directory"]

        if not directory.is_dir():
            raise CampaignError(
                f"Base corpus directory does not exist: {group['directory']}"
            )

        paths = sorted(
            p.relative_to(project_root).as_posix()
            for p in directory.rglob("*.yaml")
        )

        expected = group.get("expected_conditions")

        if expected is not None and len(paths) != expected:
            raise CampaignError(
                f"Base corpus group '{group['name']}' declares "
                f"{expected} conditions but {len(paths)} configuration files "
                "were found."
            )

        groups[group["name"]] = paths

    return groups


# ---------------------------------------------------------------------------
# Effective configuration resolution
# ---------------------------------------------------------------------------
def resolve_effective_configuration(base_config, overrides=None,
                                    project_root=PROJECT_ROOT):
    """
    Resolve one condition through the ordinary experiment argument parser.

    Returns the effective configuration exactly as an experiment run would
    build it, so its scientific identity is the identity the run will record.
    """
    argv = ["--config", str(project_root / base_config)]
    argv.extend(build_override_arguments(overrides or {}))

    buffer = io.StringIO()

    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            arguments, _ = experiment_main.parse_experiment_arguments(argv)
            return experiment_main.build_effective_configuration(arguments)
    except SystemExit as error:
        raise CampaignError(
            f"Configuration for '{base_config}' with overrides "
            f"{overrides!r} was rejected: {buffer.getvalue().strip()[-400:]}"
        ) from error


def build_override_arguments(overrides):
    """Render an override mapping as command-line arguments."""
    argv = []

    for name in sorted(overrides):
        value = overrides[name]

        if isinstance(value, bool):
            # Boolean switches are expressed by their own flags.
            argv.append(f"--{name}" if value else f"--no_{name}")
            continue

        if isinstance(value, dict):
            argv.extend([f"--{name}", json.dumps(value, sort_keys=True)])
            continue

        if isinstance(value, list):
            # An argparse nargs field expects each element as a separate
            # token.  A JSON-valued field expects one JSON string.  Use the
            # production parser to distinguish the two.
            action = _parser_action_for(name)
            if action is not None and action.nargs is not None:
                argv.append(f"--{name}")
                argv.extend(str(element) for element in value)
            else:
                argv.extend([f"--{name}", json.dumps(value, sort_keys=True)])
            continue

        argv.extend([f"--{name}", str(value)])

    return argv


_PARSER_ACTION_CACHE = None


def _parser_action_for(dest):
    """Return the argparse action for *dest*, or None if not found."""
    global _PARSER_ACTION_CACHE
    if _PARSER_ACTION_CACHE is None:
        parser = experiment_main.get_parser()
        _PARSER_ACTION_CACHE = {
            action.dest: action for action in parser._actions
        }
    return _PARSER_ACTION_CACHE.get(dest)


def scientific_fields(effective_configuration):
    """Return only the fields that participate in scientific identity."""
    classify_effective_configuration_fields(effective_configuration)

    return {
        name: value
        for name, value in effective_configuration.items()
        if name in EFFECTIVE_CONFIGURATION_SCIENTIFIC_FIELDS
    }


def changed_scientific_fields(base_configuration, condition_configuration):
    """Return the scientific fields that differ between two configurations."""
    base_fields = scientific_fields(base_configuration)
    condition_fields = scientific_fields(condition_configuration)

    missing = object()

    return sorted(
        name
        for name in set(base_fields) | set(condition_fields)
        if base_fields.get(name, missing) != condition_fields.get(name, missing)
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def enumerate_conditions(manifest, tier=None, study=None,
                         project_root=PROJECT_ROOT, resolve=True):
    """
    Return every condition the manifest declares, in a stable order.

    Ordering is by tier, study, base configuration, axis, then level, all as
    declared or sorted, so repeated calls and reordered mappings produce the
    same sequence.
    """
    conditions = []

    # Base corpus conditions are part of the core tier.
    if study is None and tier in (None, "core"):
        conditions.extend(
            enumerate_base_conditions(manifest, project_root=project_root,
                                      resolve=resolve)
        )

    for study_name, study_body in iter_studies(manifest):
        if tier is not None and study_body["tier"] != tier:
            continue

        if study is not None and study_name != study:
            continue

        conditions.extend(
            _enumerate_study(
                study_name,
                study_body,
                project_root=project_root,
                resolve=resolve,
            )
        )

    return conditions


def enumerate_base_conditions(manifest, project_root=PROJECT_ROOT,
                              resolve=True):
    """Enumerate the 234 shipped base configurations as Condition objects."""
    groups = enumerate_base_corpus(manifest, project_root=project_root)
    conditions = []

    for group_name, paths in sorted(groups.items()):
        for config_path in paths:
            scientific_sha256 = ""
            if resolve:
                configuration = resolve_effective_configuration(
                    config_path, project_root=project_root
                )
                identity = build_scientific_configuration_identity(
                    configuration
                )
                scientific_sha256 = identity["sha256"]

            conditions.append(Condition(
                tier="core",
                study="base",
                base_config=config_path,
                axis="shipped",
                level=group_name,
                overrides=(),
                output_directory=_base_output_directory(
                    group_name, config_path
                ),
                reused=False,
                identifier=f"base/{group_name}/{Path(config_path).stem}",
                scientific_sha256=scientific_sha256,
                changed_fields=(),
            ))

    return conditions


def _enumerate_study(study_name, study, project_root, resolve):
    conditions = []
    namespace = study["output_namespace"]

    for base_config in study["base_configs"]:
        base_path = project_root / base_config

        if not base_path.is_file():
            raise CampaignError(
                f"Study '{study_name}' references a configuration that does "
                f"not exist: {base_config}"
            )

        dataset = _dataset_token(base_config)
        base_configuration = (
            resolve_effective_configuration(
                base_config, project_root=project_root
            )
            if resolve
            else None
        )

        for axis in study["axes"]:
            axis_name = axis["name"]
            baseline_overrides = _baseline_overrides(axis)

            conditions.append(
                _build_condition(
                    tier=study["tier"],
                    study_name=study_name,
                    base_config=base_config,
                    dataset=dataset,
                    axis_name=axis_name,
                    level_label="baseline",
                    overrides=baseline_overrides,
                    namespace=namespace,
                    reused=True,
                    base_configuration=base_configuration,
                    project_root=project_root,
                    resolve=resolve,
                )
            )

            for level in axis["levels"]:
                conditions.append(
                    _build_condition(
                        tier=study["tier"],
                        study_name=study_name,
                        base_config=base_config,
                        dataset=dataset,
                        axis_name=axis_name,
                        level_label=level["label"],
                        overrides=_level_overrides(axis, level),
                        namespace=namespace,
                        reused=False,
                        base_configuration=base_configuration,
                        project_root=project_root,
                        resolve=resolve,
                    )
                )

    return conditions


def _baseline_overrides(axis):
    """The baseline arm applies no override; it is the shipped condition."""
    return {}


def _level_overrides(axis, level):
    if "bundle" in axis:
        return dict(level["values"])

    return {axis["field"]: level["value"]}


def _build_condition(tier, study_name, base_config, dataset, axis_name,
                     level_label, overrides, namespace, reused,
                     base_configuration, project_root, resolve):
    identifier = "/".join(
        [study_name, dataset, axis_name, level_label]
    )

    output_directory = "/".join(
        [namespace, dataset, axis_name, level_label]
    )

    scientific_sha256 = ""
    changed = ()

    if resolve:
        if reused:
            configuration = base_configuration
        else:
            configuration = resolve_effective_configuration(
                base_config,
                overrides,
                project_root=project_root,
            )
            changed = tuple(
                changed_scientific_fields(base_configuration, configuration)
            )

        identity = build_scientific_configuration_identity(configuration)
        scientific_sha256 = identity["sha256"]

    return Condition(
        tier=tier,
        study=study_name,
        base_config=base_config,
        axis=axis_name,
        level=level_label,
        overrides=tuple(sorted(overrides.items(), key=lambda kv: kv[0])),
        output_directory=output_directory,
        reused=reused,
        identifier=identifier,
        scientific_sha256=scientific_sha256,
        changed_fields=changed,
    )


def _dataset_token(base_config):
    """Use the configuration's own file stem so tokens stay unambiguous."""
    return Path(base_config).stem


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------
def count_campaign(manifest, project_root=PROJECT_ROOT, resolve=False):
    """Summarize the campaign without executing anything."""
    groups = enumerate_base_corpus(manifest, project_root=project_root)
    run_count = manifest["campaign"]["n_runs"]

    base_conditions = sum(len(paths) for paths in groups.values())

    summary = {
        "n_runs": run_count,
        "base_groups": {
            name: {
                "conditions": len(paths),
                "executions": len(paths) * run_count,
            }
            for name, paths in sorted(groups.items())
        },
        "base_conditions": base_conditions,
        "base_executions": base_conditions * run_count,
        "studies": {},
    }

    for study_name, study_body in iter_studies(manifest):
        conditions = enumerate_conditions(
            manifest,
            study=study_name,
            project_root=project_root,
            resolve=resolve,
        )
        new_conditions = [c for c in conditions if not c.reused]

        # A baseline arm appears once per axis, but every arm of one base
        # configuration refers to the same shipped condition. Report the
        # distinct base conditions being reused, not the number of arms.
        reused_base_configs = {
            condition.base_config for condition in conditions if condition.reused
        }

        summary["studies"][study_name] = {
            "tier": study_body["tier"],
            "new_conditions": len(new_conditions),
            "new_executions": len(new_conditions) * run_count,
            "reused_conditions": len(reused_base_configs),
        }

    core_studies = [
        body
        for body in summary["studies"].values()
        if body["tier"] == "core"
    ]
    optional_studies = [
        body
        for body in summary["studies"].values()
        if body["tier"] == "optional"
    ]

    summary["additional_core_conditions"] = sum(
        body["new_conditions"] for body in core_studies
    )
    summary["additional_core_executions"] = sum(
        body["new_executions"] for body in core_studies
    )
    summary["distinct_core_conditions"] = (
        base_conditions + summary["additional_core_conditions"]
    )
    summary["total_core_executions"] = (
        summary["base_executions"] + summary["additional_core_executions"]
    )
    summary["optional_conditions"] = sum(
        body["new_conditions"] for body in optional_studies
    )
    summary["optional_executions"] = sum(
        body["new_executions"] for body in optional_studies
    )
    summary["maximum_executions"] = (
        summary["total_core_executions"] + summary["optional_executions"]
    )

    return summary


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_campaign(manifest, project_root=PROJECT_ROOT):
    """
    Check the whole campaign without loading data or running an experiment.

    Every condition is resolved through the ordinary argument parser, so a
    configuration the framework would reject is reported here instead of
    hours into a campaign.
    """
    problems = []

    try:
        enumerate_base_corpus(manifest, project_root=project_root)
    except CampaignError as error:
        problems.append(str(error))

    try:
        conditions = enumerate_conditions(
            manifest, project_root=project_root, resolve=True
        )
    except CampaignError as error:
        problems.append(str(error))
        return problems

    seen_identifiers = {}
    seen_outputs = {}
    seen_identities = {}

    for condition in conditions:
        if condition.identifier in seen_identifiers:
            problems.append(
                f"Duplicate condition identifier: {condition.identifier}"
            )
        seen_identifiers[condition.identifier] = condition

        if condition.output_directory in seen_outputs:
            previous = seen_outputs[condition.output_directory]

            # A baseline arm shares its output with the shipped condition it
            # reuses, but two scheduled conditions must never collide.
            if not (condition.reused and previous.reused):
                problems.append(
                    "Duplicate output directory: "
                    f"{condition.output_directory}"
                )
        else:
            seen_outputs[condition.output_directory] = condition

        # Some shipped paper-reproduction and model-comparison YAMLs are
        # intentionally the same scientific configuration (the deepecg row is
        # present in both public corpora). They remain separate base campaign
        # conditions with separate outputs. Study additions, however, must be
        # scientifically unique so an axis cannot schedule duplicate work.
        if not condition.reused and condition.study != "base":
            previous = seen_identities.get(condition.scientific_sha256)

            if previous is not None:
                problems.append(
                    "Two scheduled conditions resolve to the same scientific "
                    f"identity: {previous.identifier} and "
                    f"{condition.identifier}"
                )
            seen_identities[condition.scientific_sha256] = condition

        problems.extend(_validate_condition_overrides(manifest, condition))
        problems.extend(_validate_run_schedule_of(manifest, condition,
                                                  project_root))

    problems.extend(
        _validate_baseline_declarations(manifest, project_root)
    )

    return problems


def _declared_axis(manifest, condition):
    for section in ("studies", "optional_studies"):
        study = (manifest.get(section) or {}).get(condition.study)

        if study is None:
            continue

        for axis in study["axes"]:
            if axis["name"] == condition.axis:
                return axis

    raise CampaignError(
        f"Condition {condition.identifier} references an undeclared axis."
    )


def _validate_condition_overrides(manifest, condition):
    """The declared axis is the only scientific field allowed to move."""
    if condition.study == "base":
        if condition.overrides or condition.changed_fields:
            return [
                f"{condition.identifier}: a shipped base condition must "
                "apply no scientific override."
            ]
        return []

    if condition.reused:
        if condition.overrides:
            return [
                f"{condition.identifier}: a reused baseline must apply no "
                "override."
            ]
        return []

    axis = _declared_axis(manifest, condition)
    allowed = set(axis["bundle"]) if "bundle" in axis else {axis["field"]}
    unexpected = sorted(set(condition.changed_fields) - allowed)

    if unexpected:
        return [
            f"{condition.identifier}: changes scientific field(s) outside its "
            f"axis: {', '.join(unexpected)}"
        ]

    if not condition.changed_fields:
        return [
            f"{condition.identifier}: applies an override that changes no "
            "scientific field."
        ]

    return []


def _validate_run_schedule_of(manifest, condition, project_root):
    """The run schedule must match the campaign declaration exactly."""
    configuration = resolve_effective_configuration(
        condition.base_config,
        condition.override_mapping(),
        project_root=project_root,
    )

    campaign = manifest["campaign"]
    problems = []

    if configuration.get("n_runs") != campaign["n_runs"]:
        problems.append(
            f"{condition.identifier}: resolves n_runs "
            f"{configuration.get('n_runs')!r}, expected {campaign['n_runs']}."
        )

    base_seed = configuration.get("seed")
    expected_seeds = list(campaign["run_seeds"])

    if [base_seed + offset for offset in range(campaign["n_runs"])] != (
        expected_seeds
    ):
        problems.append(
            f"{condition.identifier}: run seeds starting at {base_seed!r} do "
            f"not match {expected_seeds}."
        )

    if configuration.get("split_seed") is not None:
        problems.append(
            f"{condition.identifier}: split_seed is fixed at "
            f"{configuration.get('split_seed')!r}; the campaign expects it to "
            "follow each run seed."
        )

    return problems


def _validate_baseline_declarations(manifest, project_root):
    """Every declared axis baseline must match the resolved base value."""
    problems = []
    cache = {}

    for study_name, study_body in iter_studies(manifest):
        for base_config in study_body["base_configs"]:
            if base_config not in cache:
                cache[base_config] = resolve_effective_configuration(
                    base_config, project_root=project_root
                )
            base_cfg = cache[base_config]
            base_fields = scientific_fields(base_cfg)

            for axis in study_body["axes"]:
                axis_name = axis["name"]
                declared = axis["baseline"]

                if "bundle" in axis:
                    if not isinstance(declared, dict):
                        problems.append(
                            f"Study '{study_name}' axis '{axis_name}': "
                            f"baseline must be a mapping for a bundle axis."
                        )
                        continue
                    for field_name, expected_value in declared.items():
                        actual = base_fields.get(field_name)
                        if actual != expected_value:
                            problems.append(
                                f"Study '{study_name}' axis '{axis_name}' "
                                f"baseline declares {field_name}="
                                f"{expected_value!r} but the base "
                                f"configuration '{base_config}' resolves it "
                                f"to {actual!r}."
                            )
                else:
                    field_name = axis["field"]
                    actual = base_fields.get(field_name)
                    if actual != declared:
                        problems.append(
                            f"Study '{study_name}' axis '{axis_name}' "
                            f"baseline declares {field_name}={declared!r} "
                            f"but the base configuration '{base_config}' "
                            f"resolves it to {actual!r}."
                        )

    return problems


def _base_output_directory(group_name, config_path):
    """Deterministic output path for a base corpus condition."""
    return f"base/{group_name}/{Path(config_path).stem}"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def condition_result_directory(artifact_root, condition):
    return Path(artifact_root) / condition.output_directory


def is_condition_complete(artifact_root, condition,
                          expected_implementation=None,
                          project_root=PROJECT_ROOT):
    """
    Report whether a publication-eligible record already exists.

    Completion is established with the framework's own record validation, so a
    file that merely exists, or a record that fails publication eligibility or
    belongs to another configuration, does not count as complete.

    When *expected_implementation* is supplied, the record must also match
    the current implementation source identity.  This prevents a clean,
    publication-eligible record from a prior implementation version from being
    silently treated as the final-tag result.
    """
    if not condition.scientific_sha256:
        raise CampaignError(
            "Condition completeness requires a resolved scientific identity."
        )

    if expected_implementation is None:
        expected_implementation = build_experiment_implementation_identity(
            repository_root=project_root,
        )

    directory = condition_result_directory(artifact_root, condition)

    if not directory.is_dir():
        return False

    for path in sorted(directory.rglob("*.jsonl")):
        try:
            select_exact_result_record(
                path,
                result_root=directory,
                expected_scientific_sha256=condition.scientific_sha256,
                expected_implementation=expected_implementation,
                publication_mode=True,
            )
        except (ResultCollectionError, ValueError):
            continue

        return True

    return False


def build_condition_command(artifact_root, condition,
                            project_root=PROJECT_ROOT, python=None):
    """Build the argument vector that runs one condition."""
    argv = [
        python or sys.executable,
        "-m",
        "main",
        "--config",
        str(project_root / condition.base_config),
        "--results_dir",
        str(condition_result_directory(artifact_root, condition)),
    ]
    argv.extend(build_override_arguments(condition.override_mapping()))

    return argv


def run_conditions(conditions, artifact_root, project_root=PROJECT_ROOT,
                   resume=False, runner=None, stream=None):
    """Execute the scheduled conditions one at a time."""
    stream = stream or sys.stdout
    runner = runner or _default_runner
    executed = []
    skipped = []

    expected_implementation = None
    if resume:
        expected_implementation = build_experiment_implementation_identity(
            repository_root=project_root,
        )

    scheduled = [c for c in conditions if not c.reused]

    for condition in scheduled:
        if resume and is_condition_complete(
            artifact_root, condition,
            expected_implementation=expected_implementation,
            project_root=project_root,
        ):
            skipped.append(condition)
            print(f"[SKIP] {condition.identifier}", file=stream)
            continue

        command = build_condition_command(
            artifact_root, condition, project_root=project_root
        )
        print(f"[RUN ] {condition.identifier}", file=stream)
        runner(command, project_root)
        executed.append(condition)

    return executed, skipped


def _default_runner(command, project_root):
    completed = subprocess.run(command, cwd=str(project_root))

    if completed.returncode != 0:
        raise CampaignError(
            f"Condition failed with exit status {completed.returncode}: "
            + " ".join(command)
        )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate, validate and execute the declared experiment "
            "campaign."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("list", "Print every declared condition."),
        ("count", "Summarize conditions and run executions."),
        ("validate", "Check the manifest without running anything."),
        ("run", "Execute an explicitly selected tier or study."),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument(
            "--manifest",
            required=True,
            help="Path to the campaign manifest.",
        )

        if name in ("list", "run"):
            subparser.add_argument(
                "--tier",
                choices=TIERS,
                default=None,
                help="Restrict to one tier.",
            )
            subparser.add_argument(
                "--study",
                default=None,
                help="Restrict to one study.",
            )

        if name == "run":
            subparser.add_argument(
                "--artifact-root",
                required=True,
                help="Directory that will receive the campaign results.",
            )
            subparser.add_argument(
                "--resume",
                action="store_true",
                help=(
                    "Skip conditions that already have a complete, "
                    "publication-eligible result."
                ),
            )
            subparser.add_argument(
                "--dry-run",
                action="store_true",
                help="Print the conditions that would run, then stop.",
            )

    return parser


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
    except CampaignError as error:
        parser.error(str(error))

    if arguments.command == "validate":
        return _command_validate(manifest)

    if arguments.command == "count":
        return _command_count(manifest)

    if arguments.command == "list":
        return _command_list(manifest, arguments)

    return _command_run(parser, manifest, arguments)


def _command_validate(manifest):
    try:
        problems = validate_campaign(manifest)
    except CampaignError as error:
        print(f"[FAIL] {error}")
        return 1

    if problems:
        print(f"[FAIL] {len(problems)} problem(s) found:")

        for problem in problems:
            print(f"  - {problem}")

        return 1

    summary = count_campaign(manifest)
    print(
        "[OK] Campaign is valid: "
        f"{summary['distinct_core_conditions']} core conditions, "
        f"{summary['total_core_executions']} core run executions."
    )

    return 0


def _command_count(manifest):
    summary = count_campaign(manifest)
    run_count = summary["n_runs"]

    print(f"Runs per condition: {run_count}")
    print()
    print("Base corpus")

    for name, body in summary["base_groups"].items():
        print(
            f"  {name:24s} {body['conditions']:4d} conditions"
            f"  {body['executions']:5d} executions"
        )

    print(
        f"  {'total':24s} {summary['base_conditions']:4d} conditions"
        f"  {summary['base_executions']:5d} executions"
    )
    print()
    print("Studies")

    for name, body in sorted(summary["studies"].items()):
        print(
            f"  {name:32s} [{body['tier']:8s}] "
            f"{body['new_conditions']:3d} new  "
            f"{body['new_executions']:4d} executions  "
            f"{body['reused_conditions']:3d} reused"
        )

    print()
    print(
        f"Additional core conditions : {summary['additional_core_conditions']}"
    )
    print(
        f"Additional core executions : {summary['additional_core_executions']}"
    )
    print(
        f"Distinct core conditions   : {summary['distinct_core_conditions']}"
    )
    print(f"Total core executions      : {summary['total_core_executions']}")
    print(f"Optional conditions        : {summary['optional_conditions']}")
    print(f"Optional executions        : {summary['optional_executions']}")
    print(f"Maximum executions         : {summary['maximum_executions']}")

    return 0


def _command_list(manifest, arguments):
    conditions = enumerate_conditions(
        manifest,
        tier=arguments.tier,
        study=arguments.study,
        resolve=False,
    )

    for condition in conditions:
        overrides = condition.override_mapping()
        rendered = (
            ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items()))
            or "-"
        )
        print(
            f"{condition.tier:8s} {'REUSED' if condition.reused else 'NEW   '} "
            f"{condition.identifier:64s} {rendered}"
        )

    print(f"# {len(conditions)} condition(s) listed")

    return 0


def _command_run(parser, manifest, arguments):
    if arguments.tier is None and arguments.study is None:
        parser.error(
            "Refusing to run without an explicit scope. Pass --tier or "
            "--study."
        )

    core_studies = manifest.get("studies") or {}
    optional_studies = manifest.get("optional_studies") or {}
    known = set(core_studies) | set(optional_studies)

    if arguments.study is not None:

        if arguments.study not in known:
            parser.error(f"Unknown study: {arguments.study}")

        study_tier = (
            "core" if arguments.study in core_studies else "optional"
        )

        if arguments.tier is not None and arguments.tier != study_tier:
            parser.error(
                f"Study '{arguments.study}' belongs to tier '{study_tier}', "
                f"not '{arguments.tier}'."
            )

    if arguments.tier == "optional" and arguments.study is None:
        parser.error(
            "Optional execution requires an explicit --study selection."
        )

    try:
        problems = validate_campaign(manifest)
    except CampaignError as error:
        parser.error(str(error))

    if problems:
        rendered = "\n  - ".join(problems)
        parser.error(
            "Campaign validation failed before execution:\n  - " + rendered
        )

    conditions = enumerate_conditions(
        manifest,
        tier=arguments.tier,
        study=arguments.study,
        resolve=True,
    )

    scheduled = [c for c in conditions if not c.reused]

    if arguments.dry_run:
        for condition in scheduled:
            print(
                "[PLAN] "
                + " ".join(
                    build_condition_command(
                        arguments.artifact_root, condition
                    )
                )
            )

        print(f"# {len(scheduled)} condition(s) would run")

        return 0

    executed, skipped = run_conditions(
        scheduled,
        arguments.artifact_root,
        resume=arguments.resume,
    )
    print(
        f"# {len(executed)} condition(s) executed, "
        f"{len(skipped)} skipped"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
