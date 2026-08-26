import argparse
import copy
import json
import sys
import traceback

import numpy as np
import yaml
from numbers import Integral, Real
from pathlib import Path
import run
import utils
from artifact_provenance import build_data_compatibility_identity
from experiment_provenance import (
    CanonicalConfigurationError,
    build_scientific_configuration_identity,
)

# Import Dataset Loaders
from load_dataset import (
    load_ecgid_dataset, load_heartprint_dataset, load_ptb_dataset,
    load_cybhi_dataset, load_mitbih_dataset, load_nsrdb_dataset, load_ptbxl_dataset,
    BeatProvenance
)

# Import Task Runners
from run import (
    run_closed_set_identification,
    run_closed_set_verification,
    run_subject_disjoint_identification,
    run_subject_disjoint_verification,
    run_cross_session_identification,
    run_cross_session_verification,
    run_subject_disjoint_cross_session_identification,
    run_subject_disjoint_cross_session_verification
)

# Import Models
from models import (
    DeepECG, ResNet1D, RNN_ECG, HybridCNNLSTM, ECGTransformer,
    ECGXtractor, MobileNetGRU, MultiScaleCNN, SeparableResNet,
)

# Architectures selectable with --model. Adding an entry here is all that is
# required to benchmark a new model under every evaluation protocol; see
# experiments/Custom_Model.ipynb for the full contract a model must satisfy.
MODEL_REGISTRY = {
    'deepecg': DeepECG,
    'resnet1d': ResNet1D,
    'rnn': RNN_ECG,
    'hybrid': HybridCNNLSTM,
    'transformer': ECGTransformer,
    'ecgxtractor': ECGXtractor,
    'mobilenet_gru': MobileNetGRU,
    'multiscale_cnn': MultiScaleCNN,
    'separable_resnet': SeparableResNet,
}

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

CONFIG_KEY_ALIASES = {
    # Backward-compatible names used by experiment configuration files.
    "save_results_and_settings": "save_results",
    "enroll_parts": "enrol_parts",
    "num_pairs": "pair_sampling_budget",
    "sampling_mode": "pair_sampling_mode",
}


def _parse_json_mapping(value):
    """
    Parse one command-line JSON object into a Python dictionary.
    """
    if isinstance(value, dict):
        return value

    try:
        parsed_value = json.loads(value)
    except (
        TypeError,
        json.JSONDecodeError,
    ) as error:
        raise argparse.ArgumentTypeError(
            "Expected a valid JSON object."
        ) from error

    if not isinstance(
        parsed_value,
        dict,
    ):
        raise argparse.ArgumentTypeError(
            "Expected a JSON object."
        )

    return parsed_value


def _add_negation_flags(parser, option_names):
    """
    Add a ``--no_<option>`` counterpart for each boolean switch.

    Boolean options are declared with ``store_true`` and can therefore be
    enabled by a YAML configuration but never disabled from the command line.
    Each negation flag writes ``False`` to the same destination, so ordinary
    command-line precedence applies and the last flag wins.

    The negations are hidden from the help listing to keep it readable; they
    are documented in the README instead.

    Their default is suppressed deliberately. A ``store_false`` action carries
    an implicit default of ``True``, which would otherwise overwrite the real
    ``store_true`` default of ``False`` when defaults are collected by
    destination.
    """
    for option_name in option_names:
        parser.add_argument(
            f"--no_{option_name}",
            dest=option_name,
            action="store_false",
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )


def _collect_parser_defaults(parser):
    """
    Return an independent mapping of the parser's built-in defaults.
    """
    defaults = {}

    for action in parser._actions:
        if action.dest == "help":
            continue

        if action.default == argparse.SUPPRESS:
            continue

        defaults[action.dest] = copy.deepcopy(
            action.default
        )

    return defaults


def _parse_explicit_cli_arguments(parser, argv):
    """
    Parse only values explicitly supplied on the command line.

    Suppressing action defaults avoids accidental merging for ``append``
    arguments and makes precedence deterministic:

    built-in defaults < YAML values < explicit CLI values.
    """
    for action in parser._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS

    return vars(
        parser.parse_args(argv)
    )



def _normalize_explicit_cli_pair_sampling_aliases(
    values,
    parser,
):
    """
    Normalize legacy verification-pair CLI names to canonical names.

    Normalization occurs before YAML and command-line values are merged so
    explicit command-line arguments retain their normal precedence.
    """
    normalized = dict(
        values
    )

    aliases = {
        "num_pairs": "pair_sampling_budget",
        "sampling_mode": "pair_sampling_mode",
    }

    for legacy_name, canonical_name in aliases.items():
        if legacy_name not in normalized:
            continue

        legacy_value = normalized.pop(
            legacy_name
        )

        if canonical_name in normalized:
            canonical_value = normalized[
                canonical_name
            ]

            if canonical_value != legacy_value:
                parser.error(
                    "Conflicting command-line values were "
                    f"provided through '--{legacy_name}' "
                    f"and '--{canonical_name}'."
                )

            continue

        normalized[
            canonical_name
        ] = legacy_value

    return normalized


def _load_yaml_defaults(config_path, parser):
    """
    Load and validate one experiment YAML mapping.

    YAML keys are restricted to command-line argument destinations, plus
    documented backward-compatible aliases. The ``config`` key is reserved
    for the command line so configuration files cannot recursively redirect
    to another file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        parser.error(
            f"Configuration file not found: {config_path}"
        )

    print(
        "[INFO] Loading experiment parameters from YAML: "
        f"{config_path.name}"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        yaml_config = yaml.safe_load(file) or {}

    if not isinstance(yaml_config, dict):
        parser.error(
            "The YAML configuration must contain a key-value mapping."
        )

    valid_argument_names = {
        action.dest
        for action in parser._actions
        if action.dest != "help"
    }

    normalized_defaults = {}
    source_key_by_argument = {}
    unknown_keys = []

    for yaml_key, value in yaml_config.items():
        argument_name = CONFIG_KEY_ALIASES.get(
            yaml_key,
            yaml_key,
        )

        if argument_name == "config":
            parser.error(
                "The YAML key 'config' is reserved for the command line."
            )

        if argument_name not in valid_argument_names:
            unknown_keys.append(
                str(yaml_key)
            )
            continue

        if argument_name in normalized_defaults:
            previous_key = source_key_by_argument[
                argument_name
            ]
            parser.error(
                "Configuration keys "
                f"{previous_key!r} and {yaml_key!r} both map to "
                f"'{argument_name}'. Keep only one of them."
            )

        normalized_defaults[argument_name] = value
        source_key_by_argument[argument_name] = yaml_key

    if unknown_keys:
        parser.error(
            "Unknown configuration key(s): "
            + ", ".join(sorted(unknown_keys))
        )

    return normalized_defaults


def parse_experiment_arguments(argv=None):
    """
    Resolve one complete experiment configuration.

    Built-in parser defaults are applied first, values from ``--config`` are
    applied second, and arguments explicitly supplied on the CLI are applied
    last. This permits a YAML file to provide ``dataset`` and ``task`` while
    preserving conventional command-line precedence.
    """
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    parser = get_parser()
    parser_defaults = _collect_parser_defaults(
        parser
    )
    explicit_cli_values = (
        _parse_explicit_cli_arguments(
            parser,
            argv,
        )
    )

    explicit_cli_values = (
        _normalize_explicit_cli_pair_sampling_aliases(
            explicit_cli_values,
            parser,
        )
    )

    yaml_defaults = {}
    config_path = explicit_cli_values.get(
        "config"
    )

    if config_path:
        yaml_defaults = _load_yaml_defaults(
            config_path,
            parser,
        )

    effective_values = parser_defaults
    effective_values.update(
        yaml_defaults
    )
    effective_values.update(
        explicit_cli_values
    )

    args = argparse.Namespace(
        **effective_values
    )

    # Record which argument names were explicitly configured (in YAML or on
    # the command line). Some validation rules depend on whether a value was
    # supplied at all, which the merged namespace alone cannot reveal once
    # parser defaults are applied.
    args.explicitly_configured_arguments = frozenset(
        yaml_defaults
    ) | frozenset(
        explicit_cli_values
    )

    return (
        validate_experiment_arguments(
            args,
            parser,
        ),
        parser,
    )


def _normalize_minute_range(
    value,
    argument_name,
    parser,
):
    """
    Validate and normalize one continuous-recording minute range.

    A range is represented as ``(start_minute, end_minute)`` and must
    satisfy ``0 <= start_minute < end_minute``.
    """
    if not isinstance(
        value,
        (list, tuple),
    ) or len(value) != 2:
        parser.error(
            f"'{argument_name}' must contain exactly two "
            "numeric values: START_MINUTE END_MINUTE."
        )

    start_minute, end_minute = value

    for field_name, field_value in [
        ("start", start_minute),
        ("end", end_minute),
    ]:
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, Real)
            or not np.isfinite(field_value)
        ):
            parser.error(
                f"'{argument_name}' {field_name} minute must be "
                f"a finite number, received {field_value!r}."
            )

    start_minute = float(
        start_minute
    )
    end_minute = float(
        end_minute
    )

    if start_minute < 0.0:
        parser.error(
            f"'{argument_name}' start minute cannot be negative, "
            f"received {start_minute!r}."
        )

    if end_minute <= start_minute:
        parser.error(
            f"'{argument_name}' must satisfy START_MINUTE < "
            f"END_MINUTE, received "
            f"({start_minute}, {end_minute})."
        )

    return (
        start_minute,
        end_minute,
    )


def _normalize_minute_range_list(
    value,
    argument_name,
    parser,
):
    """
    Validate and normalize one or more minute ranges.

    Both a single YAML range such as ``[0, 5]`` and a collection such as
    ``[[0, 5], [10, 15]]`` are accepted.
    """
    if value is None:
        return None

    if not isinstance(
        value,
        (list, tuple),
    ) or not value:
        parser.error(
            f"'{argument_name}' must contain at least one "
            "minute range."
        )

    # Support the convenient YAML form:
    #
    # train_parts: [0, 5]
    #
    # in addition to:
    #
    # train_parts:
    #   - [0, 5]
    #   - [10, 15]
    if (
        len(value) == 2
        and all(
            isinstance(item, Real)
            and not isinstance(item, bool)
            for item in value
        )
    ):
        value = [value]

    normalized_ranges = []

    for range_index, minute_range in enumerate(
        value
    ):
        normalized_ranges.append(
            _normalize_minute_range(
                minute_range,
                (
                    f"{argument_name}"
                    f"[{range_index}]"
                ),
                parser,
            )
        )

    return normalized_ranges

# Enumerated options that may stay unset on both the command line and in YAML,
# because the dataset entry in config.yaml carries the default. Every other
# option with a fixed set of choices must hold one of them by the time the
# configuration is complete.
OPTIONAL_CHOICE_ARGUMENTS = frozenset(
    {
        "signal_type",
        "electrode_unit",
        "template_selection_method",
        "template_score_aggregation",
    }
)


def validate_experiment_arguments(args, parser):
    """
    Validate the complete experiment configuration after YAML overrides.

    Argparse validates command-line inputs before the YAML file is applied.
    This function therefore revalidates enumerated choices, numeric ranges,
    session lists, and task-specific requirements using the final effective
    configuration.
    """

    # ---------------------------------------------------------
    # 1. Require the final experiment identity
    # ---------------------------------------------------------
    missing_core_arguments = [
        argument_name
        for argument_name in (
            "dataset",
            "task",
        )
        if getattr(
            args,
            argument_name,
            None,
        ) is None
    ]

    if missing_core_arguments:
        parser.error(
            "The final experiment configuration must define: "
            + ", ".join(missing_core_arguments)
            + ". Supply them in YAML or explicitly on the CLI."
        )

    # ---------------------------------------------------------
    # 2. Revalidate argparse choices after YAML defaults
    # ---------------------------------------------------------
    for action in parser._actions:
        if action.dest == "help" or action.choices is None:
            continue

        value = getattr(args, action.dest, None)

        # Leaving one of these unset is legitimate: the dataset entry in
        # config.yaml then supplies the value, so None is not a choice
        # violation. Action defaults cannot be consulted here because
        # collecting the explicit command-line values replaces every one of
        # them with argparse.SUPPRESS.
        if value is None and action.dest in OPTIONAL_CHOICE_ARGUMENTS:
            continue

        if value not in action.choices:
            parser.error(
                f"Invalid value for '{action.dest}': {value!r}. "
                f"Expected one of: {list(action.choices)}."
            )

    # ---------------------------------------------------------
    # 3. Shared type and range validation helpers
    # ---------------------------------------------------------
    def require_integer(name, minimum=None):
        value = getattr(args, name)

        if isinstance(value, bool) or not isinstance(value, Integral):
            parser.error(
                f"'{name}' must be an integer, received {value!r}."
            )

        if minimum is not None and value < minimum:
            parser.error(
                f"'{name}' must be greater than or equal to "
                f"{minimum}, received {value!r}."
            )

    def require_real(name, minimum=None, maximum=None):
        value = getattr(args, name)

        if isinstance(value, bool) or not isinstance(value, Real):
            parser.error(
                f"'{name}' must be numeric, received {value!r}."
            )

        if not np.isfinite(value):
            parser.error(
                f"'{name}' must be finite, received {value!r}."
            )

        if minimum is not None and value < minimum:
            parser.error(
                f"'{name}' must be greater than or equal to "
                f"{minimum}, received {value!r}."
            )

        if maximum is not None and value > maximum:
            parser.error(
                f"'{name}' must be less than or equal to "
                f"{maximum}, received {value!r}."
            )

    # ---------------------------------------------------------
    # 4. Integer-valued parameters
    # ---------------------------------------------------------
    for argument_name in [
        "epochs",
        "batch_size",
        "n_runs",
        "num_beats_to_merge",
        "beat_merge_stride",
        "probe_fusion_size",
        "augmentation_copies",
    ]:
        require_integer(
            argument_name,
            minimum=1,
        )

    # Verification pair-sampling configuration.
    valid_pair_sampling_modes = (
        "all",
        "all_genuine",
        "balanced",
        "random",
    )

    legacy_sampling_mode = getattr(
        args,
        "sampling_mode",
        None,
    )

    canonical_sampling_mode = getattr(
        args,
        "pair_sampling_mode",
        None,
    )

    if (
        legacy_sampling_mode is not None
        and legacy_sampling_mode
        not in valid_pair_sampling_modes
    ):
        parser.error(
            "Invalid value for 'sampling_mode': "
            f"{legacy_sampling_mode!r}. "
            "Expected one of: "
            f"{list(valid_pair_sampling_modes)}."
        )

    if (
        canonical_sampling_mode is not None
        and canonical_sampling_mode
        not in valid_pair_sampling_modes
    ):
        parser.error(
            "Invalid value for 'pair_sampling_mode': "
            f"{canonical_sampling_mode!r}. "
            "Expected one of: "
            f"{list(valid_pair_sampling_modes)}."
        )

    if getattr(
        args,
        "num_pairs",
        None,
    ) is not None:
        require_integer(
            "num_pairs",
            minimum=1,
        )

    if getattr(
        args,
        "pair_sampling_budget",
        None,
    ) is not None:
        require_integer(
            "pair_sampling_budget",
            minimum=1,
        )

    require_integer(
        "max_impostor_pairs",
        minimum=1,
    )

    require_integer(
        "pair_sampling_seed",
        minimum=0,
    )

    try:
        (
            resolved_pair_sampling_mode,
            resolved_pair_sampling_budget,
        ) = utils._resolve_pair_sampling_arguments(
            pair_sampling_mode=getattr(
                args,
                "pair_sampling_mode",
                None,
            ),
            pair_sampling_budget=getattr(
                args,
                "pair_sampling_budget",
                None,
            ),
            sampling_mode=getattr(
                args,
                "sampling_mode",
                None,
            ),
            num_pairs=getattr(
                args,
                "num_pairs",
                None,
            ),
        )
    except ValueError as error:
        parser.error(
            str(error)
        )

    args.pair_sampling_mode = (
        resolved_pair_sampling_mode
    )

    args.pair_sampling_budget = (
        resolved_pair_sampling_budget
    )

    # Legacy names are accepted only at the input boundary. Downstream code
    # and experiment metadata use the canonical pair-sampling vocabulary.
    args.sampling_mode = None
    args.num_pairs = None

    if args.task not in {
        2,
        4,
        6,
        8,
    }:
        args.pair_sampling_mode = None
        args.pair_sampling_budget = None
        args.max_impostor_pairs = None
        args.pair_sampling_seed = None

    elif args.pair_sampling_mode == "all":
        args.pair_sampling_budget = None
        args.max_impostor_pairs = None
        args.pair_sampling_seed = None

    elif args.pair_sampling_mode == "all_genuine":
        args.pair_sampling_budget = None

    else:
        # balanced and random use a total sampling budget and a dedicated
        # sampling seed. The impostor-only cap is specific to all_genuine.
        args.max_impostor_pairs = None

    if args.beat_merge_stride > args.num_beats_to_merge:
        parser.error(
            "'beat_merge_stride' cannot exceed "
            "'num_beats_to_merge', received stride "
            f"{args.beat_merge_stride!r} for a merge width of "
            f"{args.num_beats_to_merge!r}. A larger stride would "
            "silently discard beats between consecutive samples."
        )

    try:
        utils._validate_merged_representation_partitioning(
            args.task,
            args.num_beats_to_merge,
            args.beat_merge_stride,
        )
    except ValueError as error:
        parser.error(str(error))

    require_integer("seed")

    if args.template_size is not None:
        require_integer(
            "template_size",
            minimum=1,
        )

    if args.split_seed is not None:
        # Optional: null follows the training seed; otherwise a non-negative
        # integer. require_integer rejects bool (True/False) explicitly.
        require_integer(
            "split_seed",
            minimum=0,
        )

    # ---------------------------------------------------------
    # 5. Floating-point and fraction parameters
    # ---------------------------------------------------------
    require_real(
        "lr",
        minimum=0.0,
    )

    if args.lr <= 0.0:
        parser.error(
            f"'lr' must be greater than 0, received {args.lr!r}."
        )

    require_real(
        "test_split",
        minimum=0.0,
        maximum=1.0,
    )

    if not 0.0 < args.test_split < 1.0:
        parser.error(
            "'test_split' must satisfy 0 < test_split < 1, "
            f"received {args.test_split!r}."
        )

    require_real(
        "val_split",
        minimum=0.0,
        maximum=1.0,
    )

    if args.val_split >= 1.0:
        parser.error(
            "'val_split' must satisfy 0 <= val_split < 1, "
            f"received {args.val_split!r}."
        )

    require_real(
        "sqi_threshold",
        minimum=0.0,
        maximum=1.0,
    )

    require_real(
        "sqi_keep_pct",
        minimum=0.0,
        maximum=1.0,
    )

    if args.sqi_keep_pct <= 0.0:
        parser.error(
            "'sqi_keep_pct' must satisfy 0 < sqi_keep_pct <= 1, "
            f"received {args.sqi_keep_pct!r}."
        )

    # ---------------------------------------------------------
    # 6. Training-only augmentation configuration
    # ---------------------------------------------------------
    if not isinstance(
        args.augmentation_parameters,
        dict,
    ):
        parser.error(
            "'augmentation_parameters' must be a mapping."
        )

    try:
        normalized_augmentation = (
            run._normalize_augmentation_config(
                {
                    "enabled": args.use_augmentation,
                    "method": args.augmentation_method,
                    "copies": args.augmentation_copies,
                    "parameters": (
                        args.augmentation_parameters
                    ),
                }
            )
        )
    except ValueError as error:
        parser.error(
            str(error)
        )

    args.use_augmentation = (
        normalized_augmentation["enabled"]
    )

    args.augmentation_method = (
        normalized_augmentation["method"]
    )

    args.augmentation_copies = (
        normalized_augmentation["copies"]
    )

    args.augmentation_parameters = (
        normalized_augmentation["parameters"]
    )

    if not isinstance(
        args.preprocessing_parameters,
        dict,
    ):
        parser.error(
            "'preprocessing_parameters' must be a mapping."
        )

    # ---------------------------------------------------------
    # 7. Boolean configuration values
    # ---------------------------------------------------------
    for argument_name in [
        "use_augmentation",
        "use_template",
        "use_deployment_evaluation",
        "outlier_filtering_on_train",
        "outlier_filtering_on_test",
        "save_results",
        "visualize",
        "intelligent_data_loading",
        "intelligent_weight_loading",
    ]:
        value = getattr(
            args,
            argument_name,
        )

        if not isinstance(value, bool):
            parser.error(
                f"'{argument_name}' must be Boolean, "
                f"received {value!r}."
            )

    # ---------------------------------------------------------
    # 8. Session-list validation
    # ---------------------------------------------------------
    session_arguments = [
        "train_sessions",
        "enroll_sessions",
        "probe_sessions",
        "session_for_single_session_evaluation",
    ]

    for argument_name in session_arguments:
        value = getattr(args, argument_name)

        if value is None:
            continue

        if not isinstance(value, (list, tuple)):
            parser.error(
                f"'{argument_name}' must be a list of session names, "
                f"received {value!r}."
            )

        if not value:
            parser.error(
                f"'{argument_name}' cannot be an empty list."
            )

        if not all(
            isinstance(session, str) and session.strip()
            for session in value
        ):
            parser.error(
                f"'{argument_name}' must contain only non-empty strings."
            )

    session_selector_names = (
        "train_sessions",
        "enroll_sessions",
        "probe_sessions",
        "session_for_single_session_evaluation",
    )

    if (
        args.dataset
        not in SESSION_ROUTED_DATASETS
        and any(
            getattr(
                args,
                selector_name,
            )
            is not None
            for selector_name
            in session_selector_names
        )
    ):
        parser.error(
            "Session-name selectors are supported only "
            "for HeartPrint and CYBHi."
        )

    # Probe sources must be disjoint from representation-learning sources and,
    # when a gallery is constructed, from active enrollment sources.
    if (
        args.task in [
            5,
            6,
            7,
            8,
        ]
        and args.probe_sessions
    ):
        source_sessions = set(
            args.train_sessions
            or []
        )

        if _cross_session_uses_enrollment(
            args
        ):
            source_sessions.update(
                args.enroll_sessions
                or []
            )

        shared_sessions = (
            source_sessions
            & set(
                args.probe_sessions
            )
        )

        if shared_sessions:
            parser.error(
                "Cross-session Task "
                f"{args.task} would probe session(s) "
                f"{sorted(shared_sessions)}, which also "
                "supply training or active enrollment data. "
                "Training/probe and enrollment/probe "
                "sessions must be disjoint."
            )

    # ---------------------------------------------------------
    # 9. Explicit record-role selectors
    # ---------------------------------------------------------
    record_selector_names = (
        "train_record_indices",
        "enroll_record_indices",
        "probe_record_indices",
    )

    record_selector_values = {
        selector_name: getattr(
            args,
            selector_name,
        )
        for selector_name
        in record_selector_names
    }

    supplied_record_selectors = [
        selector_name
        for (
            selector_name,
            selector_value,
        )
        in record_selector_values.items()
        if selector_value is not None
    ]

    if (
        supplied_record_selectors
        and args.dataset
        not in RECORD_ROUTED_DATASETS
    ):
        parser.error(
            "Record-index selectors are supported only "
            "for ECG-ID, PTB, and PTB-XL."
        )

    for (
        selector_name,
        selector_value,
    ) in record_selector_values.items():
        if selector_value is None:
            continue

        if (
            not isinstance(
                selector_value,
                (list, tuple),
            )
            or not selector_value
        ):
            parser.error(
                f"'{selector_name}' must contain at least "
                "one zero-based record index."
            )

        normalized_indices = []

        for index in selector_value:
            if (
                isinstance(
                    index,
                    bool,
                )
                or not isinstance(
                    index,
                    Integral,
                )
                or index < 0
            ):
                parser.error(
                    f"'{selector_name}' must contain only "
                    "non-negative integers."
                )

            normalized_indices.append(
                int(index)
            )

        if (
            len(
                set(
                    normalized_indices
                )
            )
            != len(
                normalized_indices
            )
        ):
            parser.error(
                f"'{selector_name}' cannot contain "
                "duplicate record indices."
            )

        setattr(
            args,
            selector_name,
            normalized_indices,
        )

    if (
        args.data_split_mode
        == "custom-record-split"
    ):
        if (
            args.dataset
            not in RECORD_ROUTED_DATASETS
        ):
            parser.error(
                "data_split_mode='custom-record-split' "
                "is supported only for ECG-ID, PTB, "
                "and PTB-XL."
            )

        if args.task not in [
            5,
            6,
            7,
            8,
        ]:
            parser.error(
                "data_split_mode='custom-record-split' "
                "requires Task 5, 6, 7, or 8."
            )

        if not args.train_record_indices:
            parser.error(
                "custom-record-split requires "
                "'train_record_indices'."
            )

        if not args.probe_record_indices:
            parser.error(
                "custom-record-split requires "
                "'probe_record_indices'."
            )

    elif supplied_record_selectors:
        parser.error(
            "Explicit record-role selectors require "
            "data_split_mode='custom-record-split'."
        )

    # ---------------------------------------------------------
    # 10. Continuous-recording minute ranges
    # ---------------------------------------------------------
    continuous_datasets = {
        "mitbih",
        "nsrdb",
    }

    range_arguments = {
        "single_segment_range": (
            args.single_segment_range
        ),
        "train_parts": args.train_parts,
        "enrol_parts": args.enrol_parts,
        "test_parts": args.test_parts,
    }

    if (
        args.dataset not in continuous_datasets
        and any(
            value is not None
            for value in range_arguments.values()
        )
    ):
        parser.error(
            "Minute-range arguments are supported only for "
            "the MIT-BIH and NSRDB datasets."
        )

    require_real(
        "temporal_guard_minutes",
        minimum=0.0,
    )

    if args.target_fars is not None:
        if args.task not in [2, 4, 6, 8]:
            parser.error(
                "'target_fars' applies only to the verification "
                "tasks 2, 4, 6, and 8."
            )

    if args.task in [2, 4, 6, 8]:
        from utils import _resolve_target_fars, _MANDATORY_VERIFICATION_FAR
        if args.target_fars is None:
            source = "default"
            requested = None
        elif len(args.target_fars) == 0:
            source = "explicit_empty"
            requested = []
        else:
            source = "explicit"
            requested = list(args.target_fars)

        try:
            resolved = _resolve_target_fars(args.target_fars)
        except ValueError as e:
            parser.error(str(e))

        args.target_fars = resolved
        args.far_resolution = {
            "mandatory_far": _MANDATORY_VERIFICATION_FAR,
            "requested_additional_fars": requested,
            "additional_far_source": source,
            "resolved_fars": resolved,
        }

    if (
        args.dataset not in continuous_datasets
        and args.temporal_guard_minutes != 0.0
    ):
        parser.error(
            "'temporal_guard_minutes' applies only to the "
            "MIT-BIH and NSRDB continuous recordings."
        )

    if args.single_segment_range is not None:
        args.single_segment_range = (
            _normalize_minute_range(
                args.single_segment_range,
                "single_segment_range",
                parser,
            )
        )

    args.train_parts = (
        _normalize_minute_range_list(
            args.train_parts,
            "train_parts",
            parser,
        )
    )

    args.enrol_parts = (
        _normalize_minute_range_list(
            args.enrol_parts,
            "enrol_parts",
            parser,
        )
    )

    args.test_parts = (
        _normalize_minute_range_list(
            args.test_parts,
            "test_parts",
            parser,
        )
    )

    if args.dataset in continuous_datasets:
        supported_continuous_modes = {
            "all-available",
            "single-segment",
            "custom-split",
        }

        if (
            args.data_split_mode
            not in supported_continuous_modes
        ):
            parser.error(
                f"Dataset '{args.dataset}' supports only "
                f"these data_split_mode values: "
                f"{sorted(supported_continuous_modes)}."
            )

        if (
            args.single_segment_range is not None
            and args.data_split_mode
            != "single-segment"
        ):
            parser.error(
                "'single_segment_range' can be used only when "
                "data_split_mode='single-segment'."
            )

        custom_range_values = [
            args.train_parts,
            args.enrol_parts,
            args.test_parts,
        ]

        if (
            any(
                value is not None
                for value in custom_range_values
            )
            and args.data_split_mode
            != "custom-split"
        ):
            parser.error(
                "'train_parts', 'enrol_parts', and "
                "'test_parts' can be used only when "
                "data_split_mode='custom-split'."
            )

        if args.data_split_mode == "custom-split":
            if not args.train_parts:
                parser.error(
                    "MIT-BIH and NSRDB custom splits require "
                    "at least one 'train_parts' range."
                )

            if not args.test_parts:
                parser.error(
                    "MIT-BIH and NSRDB custom splits require "
                    "at least one 'test_parts' range."
                )

        if (
            args.task in [5, 6, 7, 8]
            and args.data_split_mode
            != "custom-split"
        ):
            parser.error(
                f"Cross-session Task {args.task} on "
                f"'{args.dataset}' requires "
                "data_split_mode='custom-split' with "
                "explicit train_parts and test_parts."
            )

    # ---------------------------------------------------------
    # 11. General string validation
    # ---------------------------------------------------------
    for argument_name in [
        "data_split_mode",
        "sqi_method",
        "device",
    ]:
        value = getattr(args, argument_name)

        if not isinstance(value, str) or not value.strip():
            parser.error(
                f"'{argument_name}' must be a non-empty string."
            )

    # ---------------------------------------------------------
    # 12. Task-specific consistency
    # ---------------------------------------------------------
    if args.task in [3, 7] and not args.use_template:
        parser.error(
            f"Task {args.task} is an identification task for subjects "
            "excluded from representation learning and therefore requires "
            "'use_template: true' in YAML or '--use_template' on the CLI."
        )

    if (
        args.task in [3, 4]
        and args.use_template
        and args.template_size is None
    ):
        parser.error(
            f"Task {args.task} draws enrollment and probe samples from the "
            "same unseen subjects, so the gallery budget decides how many "
            "beats are left to probe with. Give 'template_size' an explicit "
            "positive integer; the paper configurations use 1."
        )

    # ---------------------------------------------------------
    # 13. Enrollment template mode consistency
    # ---------------------------------------------------------
    explicitly_configured = getattr(
        args,
        "explicitly_configured_arguments",
        frozenset(),
    )

    if args.enrollment_template_mode == "fusion":
        conflicting_arguments = [
            argument_name
            for argument_name in (
                "num_templates_per_identity",
                "template_selection_method",
                "template_score_aggregation",
            )
            if getattr(args, argument_name) is not None
        ]

        if conflicting_arguments:
            parser.error(
                "enrollment_template_mode='fusion' does not use "
                + ", ".join(conflicting_arguments)
                + ". Remove them or set "
                "enrollment_template_mode='multi_template'."
            )
    else:
        if "template_fusion_method" in explicitly_configured:
            parser.error(
                "enrollment_template_mode='multi_template' selects "
                "enrollment templates directly and does not use "
                "'template_fusion_method'. Remove it from the "
                "configuration."
            )

        if not args.use_template:
            parser.error(
                "enrollment_template_mode='multi_template' requires "
                "'use_template: true' in YAML or '--use_template' on "
                "the CLI."
            )

        if args.num_templates_per_identity is None:
            parser.error(
                "enrollment_template_mode='multi_template' requires "
                "'num_templates_per_identity'."
            )

        require_integer(
            "num_templates_per_identity",
            minimum=1,
        )

        if args.use_deployment_evaluation:
            parser.error(
                "enrollment_template_mode='multi_template' is not "
                "compatible with use_deployment_evaluation. Identity-level "
                "aggregation over multiple stored templates changes the "
                "score distribution a deployment threshold is calibrated "
                "on."
            )

    # ---------------------------------------------------------
    # 14. Deployment evaluation prerequisites
    # ---------------------------------------------------------
    if (
        args.use_deployment_evaluation
        and args.val_split <= 0.0
    ):
        parser.error(
            "Deployment evaluation requires val_split > 0 so that "
            "threshold calibration uses independent validation data."
        )

    return args




def build_effective_configuration(args):
    """
    Return a detached snapshot of every resolved experiment argument.

    Legacy pair-sampling aliases are omitted because their resolved
    canonical values are recorded explicitly.
    """
    excluded_aliases = {
        "num_pairs",
        "sampling_mode",
        "explicitly_configured_arguments",
    }

    return copy.deepcopy(
        dict(
            sorted(
                (
                    name,
                    value,
                )
                for name, value
                in vars(args).items()
                if name not in excluded_aliases
            )
        )
    )




SESSION_ROUTED_DATASETS = frozenset(
    {
        "heartprint",
        "cybhi",
    }
)

CONTINUOUS_ROUTED_DATASETS = frozenset(
    {
        "mitbih",
        "nsrdb",
    }
)

RECORD_ROUTED_DATASETS = frozenset(
    {
        "ecgid",
        "ptb",
        "ptbxl",
    }
)


def _cross_session_uses_enrollment(args):
    """
    Return whether the selected task consumes a gallery/enrollment role.
    """
    if args.task not in {
        5,
        6,
        7,
        8,
    }:
        return False

    if args.task == 7:
        return True

    return bool(
        args.use_template
    )


def _build_dataset_loader_kwargs(args):
    """
    Build loader arguments for the physical roles consumed by the experiment.

    Enrollment selectors are omitted when the selected evaluation branch does
    not construct a gallery. The supplied configuration is still retained in
    the effective experiment configuration.
    """
    use_enrollment = (
        _cross_session_uses_enrollment(
            args
        )
    )

    loader_kwargs = {
        "data_split_mode": (
            args.data_split_mode
        ),
        "num_beats_to_merge": (
            args.num_beats_to_merge
        ),
        "beat_merge_stride": (
            args.beat_merge_stride
        ),
        "preprocessing_config": (
            args.preprocessing_parameters
        ),
    }

    if args.dataset in SESSION_ROUTED_DATASETS:
        if args.train_sessions:
            loader_kwargs[
                "train_sessions"
            ] = args.train_sessions

        if (
            use_enrollment
            and args.enroll_sessions
        ):
            loader_kwargs[
                "enroll_sessions"
            ] = args.enroll_sessions

        if args.probe_sessions:
            loader_kwargs[
                "probe_sessions"
            ] = args.probe_sessions

        if (
            args.session_for_single_session_evaluation
        ):
            loader_kwargs[
                "session_for_single_session_evaluation"
            ] = (
                args.session_for_single_session_evaluation
            )

    if args.dataset in RECORD_ROUTED_DATASETS:
        if (
            args.train_record_indices
            is not None
        ):
            loader_kwargs[
                "train_record_indices"
            ] = args.train_record_indices

        if (
            use_enrollment
            and args.enroll_record_indices
            is not None
        ):
            loader_kwargs[
                "enroll_record_indices"
            ] = args.enroll_record_indices

        if (
            args.probe_record_indices
            is not None
        ):
            loader_kwargs[
                "probe_record_indices"
            ] = args.probe_record_indices

    if args.dataset == "ecgid":
        loader_kwargs[
            "signal_type"
        ] = args.signal_type

    if args.dataset == "cybhi":
        loader_kwargs[
            "electrode_unit"
        ] = args.electrode_unit

    if args.dataset in CONTINUOUS_ROUTED_DATASETS:
        if (
            args.single_segment_range
            is not None
        ):
            loader_kwargs[
                "single_segment_range"
            ] = (
                args.single_segment_range
            )

        if args.train_parts is not None:
            loader_kwargs[
                "train_parts"
            ] = args.train_parts

        if (
            use_enrollment
            and args.enrol_parts
            is not None
        ):
            loader_kwargs[
                "enrol_parts"
            ] = args.enrol_parts

        if args.test_parts is not None:
            loader_kwargs[
                "test_parts"
            ] = args.test_parts

        loader_kwargs[
            "temporal_guard_minutes"
        ] = args.temporal_guard_minutes

    return loader_kwargs


def _resolved_enrollment_reuses_training(
    loader,
    dataset,
):
    """
    Return whether TRAIN and ENROLL resolve to the same physical selector.

    Selector order is retained because it also determines deterministic sample
    and provenance order.
    """
    if dataset in SESSION_ROUTED_DATASETS:
        return tuple(
            getattr(
                loader,
                "train_sessions",
                None,
            )
            or ()
        ) == tuple(
            getattr(
                loader,
                "enroll_sessions",
                None,
            )
            or ()
        )

    if dataset in CONTINUOUS_ROUTED_DATASETS:
        return tuple(
            tuple(part)
            for part in (
                getattr(
                    loader,
                    "train_parts",
                    None,
                )
                or ()
            )
        ) == tuple(
            tuple(part)
            for part in (
                getattr(
                    loader,
                    "enrol_parts",
                    None,
                )
                or ()
            )
        )

    if dataset in RECORD_ROUTED_DATASETS:
        if (
            getattr(
                loader,
                "data_split_mode",
                None,
            )
            != "custom-record-split"
        ):
            return True

        return tuple(
            getattr(
                loader,
                "train_record_indices",
                None,
            )
            or ()
        ) == tuple(
            getattr(
                loader,
                "enroll_record_indices",
                None,
            )
            or ()
        )

    raise ValueError(
        "Cross-session enrollment routing is not "
        f"defined for dataset {dataset!r}."
    )


def _load_cross_session_roles(
    loader,
    use_enrollment,
    enrollment_reuses_training,
):
    """
    Load required TRAIN, ENROLL, and PROBE sources exactly once.

    When TRAIN and ENROLL resolve to the same source, enrollment is returned as
    omitted so that the runner reuses the post-training-SQI TRAIN role.
    """
    (
        x_train,
        y_train,
        provenance_train,
    ) = loader.load_session(
        "train",
        return_provenance=True,
    )

    x_enroll = None
    y_enroll = None
    provenance_enroll = None

    if (
        use_enrollment
        and not enrollment_reuses_training
    ):
        (
            x_enroll,
            y_enroll,
            provenance_enroll,
        ) = loader.load_session(
            "enrollment",
            return_provenance=True,
        )

    (
        x_probe,
        y_probe,
        provenance_probe,
    ) = loader.load_session(
        "probe",
        return_provenance=True,
    )

    return (
        x_train,
        y_train,
        provenance_train,
        x_enroll,
        y_enroll,
        provenance_enroll,
        x_probe,
        y_probe,
        provenance_probe,
    )


def _build_cross_session_cache_arrays(
    x_train,
    y_train,
    provenance_train,
    x_enroll,
    y_enroll,
    provenance_enroll,
    x_probe,
    y_probe,
    provenance_probe,
):
    """
    Build a semantic cross-session data-cache payload.
    """
    cache_arrays = {
        "x_train": x_train,
        "y_train": y_train,
        "x_probe": x_probe,
        "y_probe": y_probe,
    }

    cache_arrays.update(
        provenance_train.to_cache_dict(
            prefix="provenance_train__"
        )
    )

    cache_arrays.update(
        provenance_probe.to_cache_dict(
            prefix="provenance_probe__"
        )
    )

    if x_enroll is not None:
        cache_arrays[
            "x_enroll"
        ] = x_enroll

        cache_arrays[
            "y_enroll"
        ] = y_enroll

        cache_arrays.update(
            provenance_enroll.to_cache_dict(
                prefix="provenance_enroll__"
            )
        )

    return cache_arrays


def _restore_cross_session_cache(
    cached_data,
    require_distinct_enrollment,
):
    """
    Restore a semantic TRAIN/ENROLL/PROBE data-cache payload.

    Legacy two-role payloads do not contain these keys and therefore miss
    cleanly so that the cache is rebuilt.
    """
    if not cached_data:
        return None

    required_keys = {
        "x_train",
        "y_train",
        "x_probe",
        "y_probe",
    }

    if require_distinct_enrollment:
        required_keys.update(
            {
                "x_enroll",
                "y_enroll",
            }
        )

    if not required_keys.issubset(
        cached_data
    ):
        return None

    provenance_train = (
        BeatProvenance.from_cache_dict(
            cached_data,
            prefix="provenance_train__",
            expected_length=len(
                cached_data[
                    "y_train"
                ]
            ),
        )
    )

    provenance_probe = (
        BeatProvenance.from_cache_dict(
            cached_data,
            prefix="provenance_probe__",
            expected_length=len(
                cached_data[
                    "y_probe"
                ]
            ),
        )
    )

    if (
        provenance_train is None
        or provenance_probe is None
    ):
        return None

    x_enroll = None
    y_enroll = None
    provenance_enroll = None

    if require_distinct_enrollment:
        provenance_enroll = (
            BeatProvenance.from_cache_dict(
                cached_data,
                prefix="provenance_enroll__",
                expected_length=len(
                    cached_data[
                        "y_enroll"
                    ]
                ),
            )
        )

        if provenance_enroll is None:
            return None

        x_enroll = (
            cached_data[
                "x_enroll"
            ]
        )

        y_enroll = (
            cached_data[
                "y_enroll"
            ]
        )

    return (
        cached_data[
            "x_train"
        ],
        cached_data[
            "y_train"
        ],
        provenance_train,
        x_enroll,
        y_enroll,
        provenance_enroll,
        cached_data[
            "x_probe"
        ],
        cached_data[
            "y_probe"
        ],
        provenance_probe,
    )

def build_data_cache_config(
    args,
    loader,
    task_type,
    enrollment_consumed=None,
):
    """
    Build the deterministic identity for cached dataset arrays.

    Cross-session enrollment selectors that are not consumed by the selected
    evaluation branch do not alter the cached physical data identity.
    """
    loader_identity = (
        utils._build_loader_cache_identity(
            loader
        )
    )

    enroll_sessions = getattr(
        args,
        "enroll_sessions",
        None,
    )

    if (
        task_type == "cross_session"
        and enrollment_consumed is False
    ):
        enroll_sessions = None

    compatibility_identity = (
        build_data_compatibility_identity()
    )

    return {
        "dataset": args.dataset,
        "loader_class": (
            loader_identity[
                "loader_class"
            ]
        ),
        "task_type": task_type,
        "split_mode": (
            args.data_split_mode
        ),
        "num_beats_to_merge": (
            args.num_beats_to_merge
        ),
        "signal_type": getattr(
            loader,
            "signal_type",
            None,
        ),
        "electrode_unit": getattr(
            loader,
            "electrode_unit",
            None,
        ),
        "train_sessions": getattr(
            args,
            "train_sessions",
            None,
        ),
        "enroll_sessions": (
            enroll_sessions
        ),
        "probe_sessions": getattr(
            args,
            "probe_sessions",
            None,
        ),
        "session_for_single_session_evaluation": (
            getattr(
                args,
                "session_for_single_session_evaluation",
                None,
            )
        ),
        "dataset_config": (
            loader_identity[
                "dataset_config"
            ]
        ),
        "preprocessing": (
            loader_identity[
                "preprocessing"
            ]
        ),
        "loader_settings": (
            loader_identity[
                "settings"
            ]
        ),
        "implementation_identity": (
            compatibility_identity[
                "implementation"
            ]
        ),
        "dependency_identity": (
            compatibility_identity[
                "dependencies"
            ]
        ),
    }

def _terminate_pipeline_with_error(error):
    """
    Report an unrecoverable pipeline error and exit with a failure status.

    A nonzero exit code allows command-line scripts, CI systems, and batch
    experiment managers to distinguish failed runs from successful runs.
    """
    print(
        f"\n[CRITICAL ERROR] Pipeline Failed: {error}",
        file=sys.stderr,
    )

    traceback.print_exception(
        type(error),
        error,
        error.__traceback__,
        file=sys.stderr,
    )

    raise SystemExit(1)

def get_parser():
    # Use RawTextHelpFormatter to preserve beautiful line breaks in the help menu
    parser = argparse.ArgumentParser(
        description=(
            "========================================================================\n"
            " DEEP LEARNING ECG BIOMETRICS FRAMEWORK\n"
            "========================================================================\n"
            "Unified Command-Line Interface to evaluate ECG signals across multiple \n"
            "datasets, biometric tasks, and neural network architectures.\n\n"
            "Supported Tasks:\n"
            "  1 : Closed-Set Identification           (Intra-session, Known Subjects)\n"
            "  2 : Closed-Set Verification             (Intra-session, Known Subjects)\n"
            "  3 : Subject-Disjoint Identification     (Intra-session, Unseen Subjects)\n"
            "  4 : Subject-Disjoint Verification       (Intra-session, Unseen Subjects)\n"
            "  5 : Cross-Session Identification        (Temporal Robustness, Known)\n"
            "  6 : Cross-Session Verification          (Temporal Robustness, Known)\n"
            "  7 : Subject-Disjoint Cross-Session ID   (Ultimate Test, Unseen + Temporal)\n"
            "  8 : Subject-Disjoint Cross-Session Verif(Ultimate Test, Unseen + Temporal)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "------------------------------------------------------------------------\n"
            " EXAMPLES OF USAGE:\n"
            "------------------------------------------------------------------------\n"
            "1. Simple Closed-Set ID on ECG-ID using DeepECG Softmax:\n"
            "   python main.py --dataset ecgid --task 1 --data_split_mode single-shot-short-term --epochs 20\n\n"
            "2. Subject-Disjoint Cross-Session Verification on CYBHi with Template Matching & Outlier Filtering:\n"
            "   python main.py --dataset cybhi --task 8 --data_split_mode cross-session \\\n"
            "                  --train_sessions short-term_CI --probe_sessions short-term_A2 \\\n"
            "                  --use_template --template_size 5 --matching_method cosine \\\n"
            "                  --outlier_filtering_on_train --sqi_method kurtosis --save_results\n"
        )
    )

    # ----------------------------------------------------
    # CORE CONFIGURATION
    # ----------------------------------------------------
    core_group = parser.add_argument_group('Core Configuration')
    core_group.add_argument('--dataset', type=str, default=None,
                            choices=['ecgid', 'ptb', 'mitbih', 'nsrdb', 'ptbxl', 'heartprint', 'cybhi'],
                            help="Target database to load.")
    core_group.add_argument('--task', type=int, default=None,
                            choices=sorted(run.PUBLIC_TASK_RUNNERS),
                            help="Biometric Evaluation Task Number (1 to 8).")
    core_group.add_argument('--model', type=str, default='deepecg',
                            choices=sorted(MODEL_REGISTRY),
                            help=(
                                "Neural network architecture "
                                "(default: deepecg). Architectures marked "
                                "[lit] are re-implementations of published "
                                "ECG biometric methods:\n"
                                "  deepecg           Deep-ECG CNN [lit]\n"
                                "  ecgxtractor       autoencoder encoder [lit]\n"
                                "  mobilenet_gru     MobileNetV1 + GRU [lit]\n"
                                "  multiscale_cnn    parallel multi-scale CNN [lit]\n"
                                "  separable_resnet  residual separable CNN [lit]\n"
                                "  resnet1d          1D ResNet\n"
                                "  hybrid            CNN-LSTM\n"
                                "  transformer       ECG Transformer\n"
                                "  rnn               LSTM/GRU"
                            ))
    core_group.add_argument('--config', type=str, default=None,
                            help="Path to a YAML file providing experiment defaults. Explicit CLI values take precedence.")

    # ----------------------------------------------------
    # DATASET ROUTING & PARAMS
    # ----------------------------------------------------
    data_group = parser.add_argument_group('Dataset & Split Routing')
    data_group.add_argument('--data_split_mode', type=str, default='all-available',
                            help="Dataset parsing logic (e.g., 'single-shot-short-term', 'cross-session').")
    data_group.add_argument('--train_sessions', type=str, nargs='+',
                            help="Session tags for Training (CYBHi/HeartPrint). E.g., session1")
    data_group.add_argument('--enroll_sessions', type=str, nargs='+',
                            help="Session tags for Enrollment (CYBHi/HeartPrint).")
    data_group.add_argument('--probe_sessions', type=str, nargs='+',
                            help="Session tags for Testing/Probes (CYBHi/HeartPrint). E.g., session2")
    data_group.add_argument(
        "--train_record_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Zero-based per-subject recording positions "
            "used for representation learning when "
            "data_split_mode='custom-record-split' "
            "(ECG-ID/PTB/PTB-XL)."
        ),
    )
    data_group.add_argument(
        "--enroll_record_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Optional zero-based per-subject recording "
            "positions used for gallery enrollment when "
            "data_split_mode='custom-record-split'. "
            "Omitting this selector reuses TRAIN."
        ),
    )
    data_group.add_argument(
        "--probe_record_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help=(
            "Zero-based per-subject recording positions "
            "used for cross-session probes when "
            "data_split_mode='custom-record-split'."
        ),
    )
    data_group.add_argument('--session_for_single_session_evaluation', type=str, nargs='+',
                            help="Target session if running intra-session tasks (1-4) on multi-session datasets.")
    data_group.add_argument('--num_beats_to_merge', type=int, default=1,
                            help="Consecutive beats to fuse natively in the loader (default: 1).")
    data_group.add_argument(
        '--beat_merge_stride',
        type=int,
        default=1,
        help=(
            "Step between consecutive beat-merge windows "
            "(default: 1). The default slides one beat at a "
            "time, so merged samples share beats with their "
            "neighbours. Set it equal to --num_beats_to_merge "
            "for strictly non-overlapping samples."
        ),
    )
    data_group.add_argument(
        '--preprocessing_parameters',
        type=_parse_json_mapping,
        default={},
        metavar='JSON_OBJECT',
        help=(
            "Explicit preprocessing overrides as a JSON object. "
            "The effective canonical mapping is stored in caches and "
            "experiment logs."
        ),
    )
    data_group.add_argument('--signal_type', type=str, default=None, choices=['raw', 'filtered'],
                            help="For ECG-ID: which stored channel to read. Omit to use the "
                                 "dataset default in config.yaml, which is the raw channel.")
    data_group.add_argument('--electrode_unit', type=str, default=None,
                            choices=['8B', '85', 'both'],
                            help="For the CYBHi short-term collection: which acquiring unit "
                                 "to read. '8B' is the Ag/AgCl palm unit, '85' the electrolycra "
                                 "finger unit, and 'both' pools them. Every short-term "
                                 "acquisition was recorded by both units at once, so pooling "
                                 "mixes two electrode configurations into one identity. The "
                                 "long-term collection was acquired by a single unit and is "
                                 "unaffected. Omit to use the dataset default in config.yaml.")
    data_group.add_argument(
        "--single_segment_range",
        type=float,
        nargs=2,
        metavar=(
            "START_MINUTE",
            "END_MINUTE",
        ),
        default=None,
        help=(
            "Minute range used by MIT-BIH or NSRDB when "
            "data_split_mode='single-segment'."
        ),
    )

    data_group.add_argument(
        "--train_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=(
            "START_MINUTE",
            "END_MINUTE",
        ),
        default=None,
        help=(
            "Training minute range for MIT-BIH or NSRDB "
            "custom splits. Repeat the option to provide "
            "multiple ranges."
        ),
    )

    data_group.add_argument(
        "--enrol_parts",
        "--enroll_parts",
        dest="enrol_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=(
            "START_MINUTE",
            "END_MINUTE",
        ),
        default=None,
        help=(
            "Optional enrollment minute range for MIT-BIH "
            "or NSRDB custom splits. Both British and "
            "American spellings are accepted."
        ),
    )

    data_group.add_argument(
        "--test_parts",
        type=float,
        nargs=2,
        action="append",
        metavar=(
            "START_MINUTE",
            "END_MINUTE",
        ),
        default=None,
        help=(
            "Probe/test minute range for MIT-BIH or NSRDB "
            "custom splits. Repeat the option to provide "
            "multiple ranges."
        ),
    )

    data_group.add_argument(
        "--temporal_guard_minutes",
        type=float,
        default=0.0,
        help=(
            "Minimum separation in minutes required between the "
            "enrollment coverage and the probe coverage of a "
            "MIT-BIH or NSRDB custom split (default: 0.0). "
            "Overlapping enrollment and probe windows are always "
            "rejected; a positive value additionally rejects "
            "directly adjacent windows."
        ),
    )

    # ----------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # ----------------------------------------------------
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--epochs', type=int, default=150, help="Max epochs (default: 150).")
    train_group.add_argument('--batch_size', type=int, default=256, help="Batch size (default: 256).")
    train_group.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 0.001).")
    train_group.add_argument('--test_split', type=float, default=0.2, help="Percentage for Test set (default: 0.2).")
    train_group.add_argument('--val_split', type=float, default=0.0, help="Percentage for Validation set (default: 0.0).")
    train_group.add_argument('--seed', type=int, default=42, help="Training/general stochastic seed (model initialization, training DataLoader shuffling, augmentation, and verification-pair sampling in the balanced/random modes). When --split_seed is omitted (or YAML null), it also supplies the data-role allocation seed.")
    train_group.add_argument('--n_runs', type=int, default=1, help="Number of repeated runs using consecutive training seeds. The data-role split schedule follows the --split_seed policy.")
    train_group.add_argument('--split_seed', type=int, default=None, help="Seed for randomized data-role allocation. Omit this option (or use YAML null) to follow the per-run training seed; provide a non-negative integer to hold the randomized partition fixed across training seeds.")
    train_group.add_argument(
        '--reproducibility_mode',
        choices=['seeded', 'strict'],
        default='seeded',
        help=(
            "Reproducibility policy: 'seeded' controls framework RNGs; "
            "'strict' additionally enforces deterministic PyTorch/backend "
            "execution (default: seeded)."
        ),
    )

    # ----------------------------------------------------
    # TRAINING-ONLY DATA AUGMENTATION
    # ----------------------------------------------------
    augmentation_group = parser.add_argument_group(
        'Training-Only Data Augmentation'
    )

    augmentation_group.add_argument(
        '--use_augmentation',
        action='store_true',
        help=(
            "Append augmented copies only to the final "
            "representation-learning partition."
        ),
    )

    augmentation_group.add_argument(
        '--augmentation_method',
        type=str,
        default='gaussian',
        choices=[
            'gaussian',
            'amplitude',
            'timeshift',
            'baseline_wander',
            'time_warp',
            'cutout',
            'emg_noise',
            'istft_augment',
        ],
        help="Training augmentation method.",
    )

    augmentation_group.add_argument(
        '--augmentation_copies',
        type=int,
        default=1,
        help=(
            "Number of augmented copies appended for every "
            "original training sample."
        ),
    )

    augmentation_group.add_argument(
        '--augmentation_parameters',
        type=_parse_json_mapping,
        default={},
        metavar='JSON_OBJECT',
        help=(
            "Method-specific parameters as a JSON object. "
            "Example: '{\"std\": 0.02, "
            "\"relative\": true}'."
        ),
    )

    # ----------------------------------------------------
    # EVALUATION & TEMPLATE SETTINGS
    # ----------------------------------------------------
    eval_group = parser.add_argument_group('Evaluation & Biometric Settings')
    eval_group.add_argument('--use_template', action='store_true',
                            help="Enable Template Matching (strips Softmax). Required for Tasks 3, 4, 7, 8.")
    eval_group.add_argument(
        '--template_fusion_method',
        type=str,
        default='mean',
        choices=[
            'mean',
            'median',
            'trimmed_mean',
            'representative',
            'soft_centrality',
            'geometric_median',
            'none',
        ],
        help=(
            "Method used to aggregate enrollment embeddings into "
            "subject templates."
        ),
    )
    eval_group.add_argument('--template_size', type=int, default=None,
                            help="Max number of beats to use for enrollment (None = use all).")
    eval_group.add_argument(
        '--enrollment_template_mode',
        type=str,
        default='fusion',
        choices=['fusion', 'multi_template'],
        help=(
            "Enrollment strategy. 'fusion' (default) aggregates enrollment "
            "embeddings into templates with --template_fusion_method. "
            "'multi_template' instead stores a fixed number of "
            "representative enrollment embeddings per identity and "
            "aggregates match scores at the identity level."
        ),
    )
    eval_group.add_argument(
        '--num_templates_per_identity',
        type=int,
        default=None,
        help=(
            "Number of stored templates per enrolled identity. Required "
            "when enrollment_template_mode='multi_template'; every enrolled "
            "identity must have at least this many enrollment observations."
        ),
    )
    eval_group.add_argument(
        '--template_selection_method',
        type=str,
        default=None,
        choices=['farthest_first_cosine'],
        help=(
            "Deterministic template selection rule for "
            "enrollment_template_mode='multi_template'. Defaults to "
            "'farthest_first_cosine'."
        ),
    )
    eval_group.add_argument(
        '--template_score_aggregation',
        type=str,
        default=None,
        choices=['max'],
        help=(
            "Rule combining per-template match scores into one identity "
            "score for enrollment_template_mode='multi_template'. Defaults "
            "to 'max'."
        ),
    )
    eval_group.add_argument('--matching_method', type=str, default='cosine',
                            choices=['cosine', 'euclidean', 'manhattan', 'correlation'],
                            help="Distance metric for verification/identification.")
    eval_group.add_argument(
        '--pair_sampling_mode',
        type=str,
        default=None,
        help=(
            "Verification comparison strategy: all, all_genuine, "
            "balanced, or random. The default remains exhaustive all."
        ),
    )
    eval_group.add_argument(
        '--pair_sampling_budget',
        type=int,
        default=None,
        help=(
            "Exact number of verification decisions evaluated by balanced "
            "and random sampling. Balanced budgets must be even. Sampling "
            "uses replacement only when the requested count exceeds the "
            "relevant distinct candidate universe. Defaults to 10000 for "
            "those modes."
        ),
    )
    eval_group.add_argument(
        '--max_impostor_pairs',
        type=int,
        default=1000000,
        help=(
            "Maximum number of uniformly sampled impostor comparisons "
            "for all_genuine verification."
        ),
    )
    eval_group.add_argument(
        '--pair_sampling_seed',
        type=int,
        default=42,
        help=(
            "Dedicated random seed for stochastic verification-pair "
            "sampling."
        ),
    )
    eval_group.add_argument(
        '--num_pairs',
        type=int,
        default=None,
        help=(
            "Backward-compatible alias for --pair_sampling_budget."
        ),
    )
    eval_group.add_argument(
        '--sampling_mode',
        type=str,
        default=None,
        help=(
            "Backward-compatible alias for --pair_sampling_mode."
        ),
    )

    eval_group.add_argument(
        '--probe_fusion_size',
        type=int,
        default=1,
        help="Score-level probe fusion depth for identification and template-based verification tasks. Defaults to 1, which scores each probe beat as a single decision; values above 1 average that many probe scores from consecutive observations within one source block into one fused decision. Verification requires use_template=True and does not compose with use_deployment_evaluation."
    )

    eval_group.add_argument(
        '--use_deployment_evaluation',
        action='store_true',
        help="Calibrate a verification threshold on validation data and apply it to test data."
    )

    eval_group.add_argument(
        '--target_fars',
        type=float,
        nargs='*',
        default=None,
        metavar='FAR',
        help=(
            "Additional false-acceptance operating points reported for "
            "verification tasks, as fractions in (0, 1). "
            "If omitted, defaults to 0.1 0.01 0.0001. "
            "If passed without values, no additional points are added. "
            "The mandatory headline metric TAR@0.1%%FAR (0.001) is "
            "always unconditionally included."
        ),
    )

    # ----------------------------------------------------
    # SQI & FILTERING SETTINGS
    # ----------------------------------------------------
    sqi_group = parser.add_argument_group('Signal Quality & Filtering')
    sqi_group.add_argument('--outlier_filtering_on_train', action='store_true',
                           help="Apply SQI filtering to representation-learning data.")
    sqi_group.add_argument('--outlier_filtering_on_test', action='store_true',
                           help="Apply SQI filter to Probe/Test data.")
    sqi_group.add_argument('--sqi_method', type=str, default='kurtosis',
                           help="Method to evaluate signal quality (e.g., 'kurtosis').")
    sqi_group.add_argument('--sqi_threshold', type=float, default=0.05,
                           help="Absolute minimum quality score to survive (0.0 to 1.0).")
    sqi_group.add_argument('--sqi_keep_pct', type=float, default=0.8,
                           help="Percentage of the best beats to keep per subject (default: 0.8 = 80%%).")

    # ----------------------------------------------------
    # LOGGING & MISC
    # ----------------------------------------------------
    misc_group = parser.add_argument_group('Logging & Misc')
    misc_group.add_argument('--save_results', action='store_true',
                            help="If set, writes experiment settings and results to the results folder.")
    misc_group.add_argument('--visualize', action='store_true',
                            help="If set, plots t-SNE / CMC / Confusion Matrices.")
    misc_group.add_argument('--device', type=str, default='auto',
                            help="Device to use ('cuda', 'cpu', or 'auto').")
    misc_group.add_argument('--intelligent_data_loading', action='store_true',
                            help="If set, saves/loads precomputed data arrays based on hyperparameters.")
    misc_group.add_argument('--intelligent_weight_loading', action='store_true',
                            help="If set, saves/loads pre-trained model weights based on hyperparameters.")

    # A YAML file can switch any store_true option on, so each one needs a
    # command-line way to switch it back off. Without these, a configuration
    # that enables caching could not be overridden without editing the file.
    _add_negation_flags(
        parser,
        [
            'save_results',
            'visualize',
            'use_template',
            'use_augmentation',
            'use_deployment_evaluation',
            'outlier_filtering_on_train',
            'outlier_filtering_on_test',
            'intelligent_data_loading',
            'intelligent_weight_loading',
        ],
    )
    misc_group.add_argument(
        '--cache_dir',
        type=str,
        default=utils.DEFAULT_CACHE_DIR,
        help=(
            "Directory for preprocessed arrays and trained model "
            "weights. Relative paths are resolved from the repository."
        ),
    )
    misc_group.add_argument(
        '--results_dir',
        type=str,
        default=utils.DEFAULT_RESULTS_DIR,
        help=(
            "Directory for experiment outputs. Relative paths are "
            "resolved from the repository."
        ),
    )
    misc_group.add_argument(
        '--campaign_id',
        type=str,
        default=None,
        help=(
            "Optional administrative campaign identifier recorded "
            "separately from scientific configuration identity."
        ),
    )
    misc_group.add_argument(
        '--smoke_run',
        action='store_true',
        help=(
            "Mark the execution as a reduced smoke/test run in result "
            "provenance."
        ),
    )

    return parser

def main():
    args, parser = parse_experiment_arguments()

    # Strict CUDA setup must precede timer/profiling availability queries and
    # every other operation that may initialize CUDA. Direct runner APIs apply
    # this same preparation independently.
    args.reproducibility_mode = (
        utils._prepare_reproducibility_backend(
            args.reproducibility_mode,
            args.device,
        )
    )

    # ==========================================
    # 0. EFFECTIVE CONFIGURATION
    # ==========================================
    effective_configuration = (
        build_effective_configuration(
            args
        )
    )

    # Fail fast on a configuration that could not become a canonical
    # scientific identity (an unclassified field, or a non-finite/
    # unsupported value reachable only through a JSON-mapping argument)
    # before any data loading or training, rather than discovering it only
    # once a completed run reaches result logging.
    try:
        build_scientific_configuration_identity(
            effective_configuration
        )
    except CanonicalConfigurationError as error:
        parser.error(str(error))

    # Measure the complete pipeline, including data loading,
    # training, evaluation, and saved-result generation.
    run.start_experiment_timer(device=args.device)

    # ==========================================
    # 1. MODEL SELECTION
    # ==========================================
    selected_model_class = MODEL_REGISTRY[args.model.lower()]

    # ==========================================
    # 2. DATASET INSTANTIATION
    # ==========================================
    # Route only the physical roles consumed by this experiment.
    loader_kwargs = (
        _build_dataset_loader_kwargs(
            args
        )
    )

    print(f"\n[INFO] Initializing {args.dataset.upper()} Dataset...")

    if args.dataset == 'ecgid': loader = load_ecgid_dataset(**loader_kwargs)
    elif args.dataset == 'ptb': loader = load_ptb_dataset(**loader_kwargs)
    elif args.dataset == 'mitbih': loader = load_mitbih_dataset(**loader_kwargs)
    elif args.dataset == 'nsrdb': loader = load_nsrdb_dataset(**loader_kwargs)
    elif args.dataset == 'ptbxl': loader = load_ptbxl_dataset(**loader_kwargs)
    elif args.dataset == 'heartprint': loader = load_heartprint_dataset(**loader_kwargs)
    elif args.dataset == 'cybhi': loader = load_cybhi_dataset(**loader_kwargs)
    else:
        print(f"[ERROR] Unsupported dataset: {args.dataset}")
        sys.exit(1)

    loader.cache_dir = args.cache_dir
    loader.results_dir = args.results_dir

    loader.effective_experiment_configuration = (
        effective_configuration
    )
    loader.result_provenance_context = {
        "campaign_id": args.campaign_id,
        "smoke_run": args.smoke_run,
    }

    cross_session_uses_enrollment = False
    cross_session_enrollment_reuses_training = False

    if args.task in [
        5,
        6,
        7,
        8,
    ]:
        cross_session_uses_enrollment = (
            _cross_session_uses_enrollment(
                args
            )
        )

        if cross_session_uses_enrollment:
            cross_session_enrollment_reuses_training = (
                _resolved_enrollment_reuses_training(
                    loader,
                    args.dataset,
                )
            )

    # ==========================================
    # 3. DATA EXTRACTION LOGIC
    # ==========================================
    data_preparation_started = (
        run._start_runtime_stage()
    )

    if args.intelligent_data_loading:
        from utils import CacheManager
        cache = CacheManager(
            base_dir=args.cache_dir
        )

        # Tasks 1 to 4: Intra-Session
        if args.task in [1, 2, 3, 4]:
            data_config = build_data_cache_config(
                args,
                loader,
                task_type="intra_session",
            )
            cached_data, uid = run._timed_runtime_call(
                "Data Cache Read",
                cache.get_data_cache,
                data_config,
            )

            cached_provenance = None
            if cached_data and "x" in cached_data and "y" in cached_data:
                cached_provenance = BeatProvenance.from_cache_dict(
                    cached_data,
                    expected_length=len(cached_data["y"]),
                )

            if cached_provenance is not None:
                print(f"[INFO] Loaded precomputed data from cache (Hash: {uid})")
                x, y = cached_data["x"], cached_data["y"]
                provenance = cached_provenance
            else:
                if args.dataset in ["cybhi", "heartprint"]:
                    x, y, provenance = loader.load_session(
                        "train", return_provenance=True
                    )
                elif args.data_split_mode in [
                    "all-available",
                    "single-session",
                    "single-segment",
                ]:
                    x, y, provenance = loader.load_all_data(
                        return_provenance=True
                    )
                else:
                    x, y, provenance = loader.load_session(
                        "train", return_provenance=True
                    )
                cache_arrays = {"x": x, "y": y}
                cache_arrays.update(provenance.to_cache_dict())
                run._timed_runtime_call(
                    "Data Cache Write",
                    cache.save_data_cache,
                    cache_arrays,
                    data_config,
                    uid,
                )

            print(f"\n[INFO] Data Loaded: X={x.shape}, Y={y.shape}")
            if x.shape[0] == 0:
                print("[ERROR] No data returned from loader. Check your session configs.")
                sys.exit(1)

        # Tasks 5 to 8: Cross-Session
        elif args.task in [5, 6, 7, 8]:
            data_config = build_data_cache_config(
                args,
                loader,
                task_type="cross_session",
                enrollment_consumed=(
                    cross_session_uses_enrollment
                ),
            )

            cached_data, uid = (
                run._timed_runtime_call(
                    "Data Cache Read",
                    cache.get_data_cache,
                    data_config,
                )
            )

            require_distinct_enrollment = (
                cross_session_uses_enrollment
                and not (
                    cross_session_enrollment_reuses_training
                )
            )

            restored_roles = (
                _restore_cross_session_cache(
                    cached_data,
                    require_distinct_enrollment=(
                        require_distinct_enrollment
                    ),
                )
            )

            if restored_roles is not None:
                (
                    x_train,
                    y_train,
                    provenance_train,
                    x_enroll,
                    y_enroll,
                    provenance_enroll,
                    x_probe,
                    y_probe,
                    provenance_probe,
                ) = restored_roles

                print(
                    "[INFO] Loaded precomputed cross-session "
                    f"data from cache (Hash: {uid})"
                )

            else:
                (
                    x_train,
                    y_train,
                    provenance_train,
                    x_enroll,
                    y_enroll,
                    provenance_enroll,
                    x_probe,
                    y_probe,
                    provenance_probe,
                ) = _load_cross_session_roles(
                    loader,
                    use_enrollment=(
                        cross_session_uses_enrollment
                    ),
                    enrollment_reuses_training=(
                        cross_session_enrollment_reuses_training
                    ),
                )

                cache_arrays = (
                    _build_cross_session_cache_arrays(
                        x_train,
                        y_train,
                        provenance_train,
                        x_enroll,
                        y_enroll,
                        provenance_enroll,
                        x_probe,
                        y_probe,
                        provenance_probe,
                    )
                )

                run._timed_runtime_call(
                    "Data Cache Write",
                    cache.save_data_cache,
                    cache_arrays,
                    data_config,
                    uid,
                )

            print(
                "\n[INFO] Training Loaded: "
                f"X={x_train.shape}, Y={y_train.shape}"
            )

            if x_enroll is not None:
                print(
                    "[INFO] Enrollment Loaded: "
                    f"X={x_enroll.shape}, "
                    f"Y={y_enroll.shape}"
                )

            elif (
                cross_session_uses_enrollment
                and (
                    cross_session_enrollment_reuses_training
                )
            ):
                print(
                    "[INFO] Enrollment reuses the Training "
                    "source after training-only filtering."
                )

            else:
                print(
                    "[INFO] Enrollment is not consumed by "
                    "the selected evaluation branch."
                )

            print(
                "[INFO] Probe Loaded: "
                f"X={x_probe.shape}, Y={y_probe.shape}"
            )

            if (
                x_train.shape[0] == 0
                or x_probe.shape[0] == 0
                or (
                    require_distinct_enrollment
                    and (
                        x_enroll is None
                        or x_enroll.shape[0] == 0
                    )
                )
            ):
                print(
                    "[ERROR] One or more required "
                    "cross-session roles are empty. "
                    "Check the configured selectors."
                )
                sys.exit(1)

    else:
        # Tasks 1 to 4: Intra-Session or Single Array operations
        if args.task in [1, 2, 3, 4]:
            if args.dataset in [
                "cybhi",
                "heartprint",
            ]:
                x, y, provenance = loader.load_session(
                    "train", return_provenance=True
                )
            else:
                if args.data_split_mode in [
                    "all-available",
                    "single-session",
                    "single-segment",
                ]:
                    x, y, provenance = loader.load_all_data(
                        return_provenance=True
                    )
                else:
                    x, y, provenance = loader.load_session(
                        "train", return_provenance=True
                    )

            print(f"\n[INFO] Data Loaded: X={x.shape}, Y={y.shape}")
            if x.shape[0] == 0:
                print("[ERROR] No data returned from loader. Check your session configs.")
                sys.exit(1)

        # Tasks 5 to 8: Cross-Session
        elif args.task in [5, 6, 7, 8]:
            (
                x_train,
                y_train,
                provenance_train,
                x_enroll,
                y_enroll,
                provenance_enroll,
                x_probe,
                y_probe,
                provenance_probe,
            ) = _load_cross_session_roles(
                loader,
                use_enrollment=(
                    cross_session_uses_enrollment
                ),
                enrollment_reuses_training=(
                    cross_session_enrollment_reuses_training
                ),
            )

            require_distinct_enrollment = (
                cross_session_uses_enrollment
                and not (
                    cross_session_enrollment_reuses_training
                )
            )

            print(
                "\n[INFO] Training Loaded: "
                f"X={x_train.shape}, Y={y_train.shape}"
            )

            if x_enroll is not None:
                print(
                    "[INFO] Enrollment Loaded: "
                    f"X={x_enroll.shape}, "
                    f"Y={y_enroll.shape}"
                )

            elif (
                cross_session_uses_enrollment
                and (
                    cross_session_enrollment_reuses_training
                )
            ):
                print(
                    "[INFO] Enrollment reuses the Training "
                    "source after training-only filtering."
                )

            else:
                print(
                    "[INFO] Enrollment is not consumed by "
                    "the selected evaluation branch."
                )

            print(
                "[INFO] Probe Loaded: "
                f"X={x_probe.shape}, Y={y_probe.shape}"
            )

            if (
                x_train.shape[0] == 0
                or x_probe.shape[0] == 0
                or (
                    require_distinct_enrollment
                    and (
                        x_enroll is None
                        or x_enroll.shape[0] == 0
                    )
                )
            ):
                print(
                    "[ERROR] One or more required "
                    "cross-session roles are empty. "
                    "Check the configured selectors."
                )
                sys.exit(1)


    run._record_runtime_stage(
        "Data Preparation (inclusive)",
        data_preparation_started,
    )

    # ==========================================
    # 4. SHARED EXECUTION ARGUMENTS
    # ==========================================
    # Arguments common to ALL 8 tasks
    common_args = {
        'model_class': selected_model_class,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'val_split': args.val_split,
        'seed': args.seed,
        'split_seed': args.split_seed,
        'reproducibility_mode': args.reproducibility_mode,
        'n_runs': args.n_runs,
        'device': args.device,
        'visualize': args.visualize,
        'use_template': args.use_template,
        'template_fusion_method': args.template_fusion_method,
        'template_size': args.template_size,
        'matching_method': args.matching_method,
        'outlier_filtering_on_train': args.outlier_filtering_on_train,
        'outlier_filtering_on_test': args.outlier_filtering_on_test,
        'sqi_threshold': args.sqi_threshold,
        'sqi_keep_pct': args.sqi_keep_pct,
        'save_results_and_settings': args.save_results,
        'loader': loader,
        'intelligent_weight_loading': args.intelligent_weight_loading,
        'augmentation_config': {
            'enabled': args.use_augmentation,
            'method': args.augmentation_method,
            'copies': args.augmentation_copies,
            'parameters': args.augmentation_parameters,
        },
    }

    common_args['enrollment_template_mode'] = args.enrollment_template_mode

    if args.enrollment_template_mode == 'multi_template':
        # Multi-template enrollment replaces template fusion entirely: the
        # runners reject an explicit fusion method in this mode, so the
        # argument is withheld rather than forwarded at its parser default.
        common_args.pop('template_fusion_method')
        common_args['num_templates_per_identity'] = args.num_templates_per_identity
        common_args['template_selection_method'] = args.template_selection_method
        common_args['template_score_aggregation'] = args.template_score_aggregation

    # ==========================================
    # 5. EXECUTE THE SELECTED TASK
    # ==========================================
    print("=" * 70)
    print(f" EXECUTING TASK {args.task} ON {args.dataset.upper()} USING {args.model.upper()}")
    print("=" * 70)

    try:
        # TASK 1: Closed-Set Identification
        if args.task == 1:
            run_closed_set_identification(
                x, y,
                test_split=args.test_split,
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                provenance=provenance,
                **common_args
            )

        # TASK 2: Verification
        elif args.task == 2:
            run_closed_set_verification(
                x, y,
                test_split=args.test_split,
                pair_sampling_budget=args.pair_sampling_budget,
                pair_sampling_mode=args.pair_sampling_mode,
                max_impostor_pairs=args.max_impostor_pairs,
                pair_sampling_seed=args.pair_sampling_seed,
                use_deployment_evaluation=args.use_deployment_evaluation,
                target_fars=args.target_fars,
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                provenance=provenance,
                **common_args
            )

        # TASK 3: Subject-Disjoint Identification
        elif args.task == 3:
            run_subject_disjoint_identification(
                x, y,
                test_split=args.test_split,
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                provenance=provenance,
                **common_args
            )

        # TASK 4: Subject-Disjoint Verification
        elif args.task == 4:
            run_subject_disjoint_verification(
                x, y,
                test_split=args.test_split,
                pair_sampling_budget=args.pair_sampling_budget,
                pair_sampling_mode=args.pair_sampling_mode,
                max_impostor_pairs=args.max_impostor_pairs,
                pair_sampling_seed=args.pair_sampling_seed,
                use_deployment_evaluation=args.use_deployment_evaluation,
                target_fars=args.target_fars,
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                provenance=provenance,
                **common_args
            )

        # TASK 5: Cross-Session Identification
        elif args.task == 5:
            run_cross_session_identification(
                x_train, y_train, x_probe, y_probe,
                probe_fusion_size=args.probe_fusion_size,
                sqi_train=args.sqi_method,
                sqi_test=args.sqi_method,
                provenance_s1=provenance_train,
                provenance_s2=provenance_probe,
                x_enroll=x_enroll,
                y_enroll=y_enroll,
                provenance_enroll=provenance_enroll,
                **common_args
            )

        # TASK 6: Cross-Session Verification
        elif args.task == 6:
            run_cross_session_verification(
                x_train, y_train, x_probe, y_probe,
                pair_sampling_budget=args.pair_sampling_budget,
                pair_sampling_mode=args.pair_sampling_mode,
                max_impostor_pairs=args.max_impostor_pairs,
                pair_sampling_seed=args.pair_sampling_seed,
                use_deployment_evaluation=args.use_deployment_evaluation,
                target_fars=args.target_fars,
                probe_fusion_size=args.probe_fusion_size,
                sqi_train=args.sqi_method,
                sqi_test=args.sqi_method,
                provenance_s1=provenance_train,
                provenance_s2=provenance_probe,
                x_enroll=x_enroll,
                y_enroll=y_enroll,
                provenance_enroll=provenance_enroll,
                **common_args
            )

        # TASK 7: Subject-Disjoint Cross-Session ID
        elif args.task == 7:
            run_subject_disjoint_cross_session_identification(
                x_train, y_train, x_probe, y_probe,
                test_split=args.test_split,
                probe_fusion_size=args.probe_fusion_size,
                sqi_s1=args.sqi_method,
                sqi_s2=args.sqi_method,
                provenance_s1=provenance_train,
                provenance_s2=provenance_probe,
                x_enroll=x_enroll,
                y_enroll=y_enroll,
                provenance_enroll=provenance_enroll,
                **common_args
            )

        # TASK 8: Subject-Disjoint Cross-Session Verification
        elif args.task == 8:
            run_subject_disjoint_cross_session_verification(
                x_train, y_train, x_probe, y_probe,
                test_split=args.test_split,
                pair_sampling_budget=args.pair_sampling_budget,
                pair_sampling_mode=args.pair_sampling_mode,
                max_impostor_pairs=args.max_impostor_pairs,
                pair_sampling_seed=args.pair_sampling_seed,
                use_deployment_evaluation=args.use_deployment_evaluation,
                target_fars=args.target_fars,
                probe_fusion_size=args.probe_fusion_size,
                sqi_s1=args.sqi_method,
                sqi_s2=args.sqi_method,
                provenance_s1=provenance_train,
                provenance_s2=provenance_probe,
                x_enroll=x_enroll,
                y_enroll=y_enroll,
                provenance_enroll=provenance_enroll,
                **common_args
            )

        print("\n[SUCCESS] Pipeline execution complete.")

    except Exception as error:
        _terminate_pipeline_with_error(error)

if __name__ == "__main__":
    main()
