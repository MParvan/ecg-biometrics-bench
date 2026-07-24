# run.py
# -----------------------------------------------------------------------------
# UNIFIED TRAINING & EVALUATION UTILITY FOR ECG BIOMETRICS
# -----------------------------------------------------------------------------
# This module handles the core deep learning and biometric evaluation logic.
# It features advanced training loops including dynamic Learning Rate rollback,
# strict temporal isolation, and a Composite Validation Metric (CE Loss + EER) 
# to optimize Subject-Disjoint generalization.
#
# SUPPORTED TASKS:
#   1. Closed-Set Identification           (Intra-session, Known Subjects)
#   2. Closed-Set Verification             (Intra-session, Known Subjects)
#   3. Subject-Disjoint Identification     (Intra-session, Unseen Subjects)
#   4. Subject-Disjoint Verification       (Intra-session, Unseen Subjects)
#   5. Cross-Session Identification        (Temporal Robustness, Known Subjects)
#   6. Cross-Session Verification          (Temporal Robustness, Known Subjects)
#   7. Subject-Disjoint Cross-Session ID   (Ultimate Test: Unseen + Temporal)
#   8. Subject-Disjoint Cross-Session Verif(Ultimate Test: Unseen + Temporal)
#
# METRICS:
#   - Identification: Rank-1 Accuracy, Rank-5 Accuracy (with Score-Level Fusion)
#   - Verification:   EER, AUC, d-prime, TAR @ 0.1% FAR
# -----------------------------------------------------------------------------

import numpy as np
import random
import collections
import copy
from typing import Dict, Any, Optional, Tuple, List, Union

import platform
import sys
from importlib import metadata

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from utils import (
    _apply_score_fusion, _make_loader, _encode_labels, _get_device, _set_seed,
    _apply_outlier_filter, _compute_sqi, _compute_score_matrix,
    _get_embeddings, _create_templates, _generate_pairs,
    _find_optimal_threshold, _evaluate_with_global_threshold, _summarize_verification_pairs,
    _compute_metrics_identification, _compute_metrics_verification,
    _run_training_loop, _run_train_loop_unseen_subjects, _train_epoch, _detect_channels
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import datetime
from pathlib import Path

from visualizations import Visualizer

# =============================================================================
# REPRODUCIBILITY ENVIRONMENT
# =============================================================================

def _get_installed_package_version(distribution_name):
    """
    Return the installed version of a Python distribution.

    Missing optional dependencies are reported explicitly rather than
    causing experiment logging to fail.
    """
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _collect_software_environment():
    """
    Collect the software and hardware environment used by an experiment.
    """
    environment = {
        "Python": platform.python_version(),
        "Operating System": platform.platform(),
        "PyTorch": str(torch.__version__),
        "CUDA Available": bool(torch.cuda.is_available()),
        "CUDA Runtime": (
            str(torch.version.cuda)
            if torch.version.cuda is not None
            else "not available"
        ),
        "NumPy": _get_installed_package_version("numpy"),
        "SciPy": _get_installed_package_version("scipy"),
        "scikit-learn": _get_installed_package_version(
            "scikit-learn"
        ),
        "pandas": _get_installed_package_version("pandas"),
        "NeuroKit2": _get_installed_package_version(
            "neurokit2"
        ),
        "WFDB": _get_installed_package_version("wfdb"),
        "PyYAML": _get_installed_package_version("PyYAML"),
    }

    if torch.cuda.is_available():
        try:
            environment["CUDA Device"] = (
                torch.cuda.get_device_name(0)
            )
        except Exception:
            environment["CUDA Device"] = "unavailable"

    return environment

# =============================================================================
# AUTOMATED EXPERIMENT LOGGER
# =============================================================================
def _log_experiment_results(task_name, metrics_dict, data_stats, hyperparams, loader=None):
    """
    Dynamically writes experiment configurations and results to a text file.
    Intelligently extracts parameters directly from the dataset loader object.
    Automatically categorizes and saves the logs into task-specific .txt files.
    """
    dataset_name = "unknown_dataset"
    dataset_kwargs = {}
    
    if loader is not None:
        # 1. Extract Dataset Name from cfg dict
        if hasattr(loader, 'cfg') and 'root_dir' in loader.cfg:
            dataset_name = str(loader.cfg['root_dir']).lower()
            
        # 2. Extract Preprocessing Params (Merge defaults with user overrides)
        default_prep = loader.cfg.get('preprocessing', {}) if hasattr(loader, 'cfg') else {}
        user_prep = getattr(loader, 'prep_params', {})
        
        # This guarantees user-defined params overwrite the defaults
        dataset_kwargs.update(default_prep)
        dataset_kwargs.update(user_prep) 
        
        # 3. Extract other useful loader attributes dynamically
        # We ignore backend objects, large paths, and redundant dictionaries
        ignore_keys = ['preprocessor', 'cfg', 'data_root', 'dataset_root', 
                       'zip_path', 'url', 'prep_params', 'cleanup_zip']
        
        for k, v in vars(loader).items():
            if k not in ignore_keys and not k.startswith('_'):
                dataset_kwargs[k] = v

    # 4. Ensure Results Directory exists: ./results/dataset_name/
    results_dir = Path("results") / dataset_name.replace(" ", "_")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. TARGET FILE: Dynamically name the file based on the Task Name
    # (e.g., "Closed-Set Identification" -> "Closed-Set_Identification.txt")
    safe_task_name = str(task_name).replace(" ", "_").replace("/", "_").replace("\\", "_")
    log_file = results_dir / f"{safe_task_name}.txt"

    software_environment = _collect_software_environment()
    
    # 6. Format and Append
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"EXPERIMENT TIME : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TASK            : {task_name}\n")
        f.write(f"DATASET         : {dataset_name}\n")
        f.write(f"{'-'*70}\n")
        
        f.write("[DATA STATISTICS]\n")
        for k, v in data_stats.items():
            f.write(f"  {k:<28}: {v}\n")
        f.write(f"{'-'*70}\n")
        
        f.write("[MODEL HYPERPARAMETERS]\n")
        for k, v in hyperparams.items():
            f.write(f"  {k:<28}: {v}\n")
            
        if dataset_kwargs:
            f.write(f"{'-'*70}\n")
            f.write("[DATASET & PREPROCESSING SETTINGS]\n")
            for k, v in dataset_kwargs.items():
                f.write(f"  {k:<28}: {v}\n")

        f.write(f"{'-'*70}\n")
        f.write("[SOFTWARE & HARDWARE ENVIRONMENT]\n")

        for key, value in software_environment.items():
            f.write(f"  {key:<28}: {value}\n")
                    
        f.write(f"{'-'*70}\n")
        f.write("[RESULTS]\n")
        for k, v in metrics_dict.items():
            if isinstance(v, float):
                f.write(f"  {k:<28}: {v:.4f}\n")
            else:
                f.write(f"  {k:<28}: {v}\n")
        f.write(f"{'='*70}\n")
    
    print(f"\n[INFO] Experiment settings and results successfully saved to: {log_file}")

# =============================================================================
# EVALUATION CONFIGURATION VALIDATION
# =============================================================================
def _validate_deployment_evaluation(
    use_deployment_evaluation,
    val_split,
    task_name,
):
    """
    Require an independent validation partition for threshold calibration.

    Deployment evaluation estimates a decision threshold on validation data
    and then freezes that threshold for final test evaluation. Calibrating on
    training data would produce optimistically biased deployment results.
    """
    if not use_deployment_evaluation:
        return

    try:
        validation_fraction = float(val_split)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{task_name}: val_split must be numeric when "
            "deployment evaluation is enabled."
        ) from error

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"{task_name}: deployment evaluation requires "
            "0 < val_split < 1. Training-data threshold calibration "
            "is not permitted."
        )

# =============================================================================
# MULTI-RUN ARGUMENT HANDLING
# =============================================================================

def _prepare_multi_run_arguments(local_arguments):
    """
    Prepare arguments for a recursive single-seed experiment run.

    """
    call_args = dict(local_arguments)

    internal_keys = {
        "data_stats",
        "hyperparams",
        "call_args",
    }

    for key in internal_keys:
        call_args.pop(key, None)

    call_args.update(
        {
            "n_runs": 1,
            "_return_stats": True,
            "save_results_and_settings": False,
        }
    )

    return call_args

# =============================================================================
# WEIGHT CACHE CONFIGURATION
# =============================================================================

def _build_weight_cache_config(loader, training_config):
    """
    Add dataset and loader identity to a model-weight cache configuration.

    Model weights are reusable only when both the training hyperparameters
    and the effective dataset and preprocessing configuration are equivalent.
    """
    complete_config = dict(training_config)

    if loader is None:
        complete_config["loader_identity"] = None
        return complete_config

    loader_cfg = getattr(loader, "cfg", {})
    effective_preprocessing = {}

    if isinstance(loader_cfg, dict):
        configured_preprocessing = loader_cfg.get(
            "preprocessing",
            {},
        )

        if isinstance(configured_preprocessing, dict):
            effective_preprocessing.update(
                configured_preprocessing
            )

    preprocessing_overrides = getattr(
        loader,
        "prep_params",
        {},
    )

    if isinstance(preprocessing_overrides, dict):
        effective_preprocessing.update(
            preprocessing_overrides
        )

    loader_settings = {}

    cache_relevant_attributes = [
        "data_split_mode",
        "num_beats_to_merge",
        "beat_merge_method",
        "signal_type",
        "train_sessions",
        "enroll_sessions",
        "enrol_sessions",
        "probe_sessions",
        "session_for_single_session_evaluation",
        "leads",
        "single_segment_range",
        "train_parts",
        "enrol_parts",
        "enroll_parts",
        "test_parts",
    ]

    for attribute_name in cache_relevant_attributes:
        if hasattr(loader, attribute_name):
            loader_settings[attribute_name] = getattr(
                loader,
                attribute_name,
            )

    complete_config["loader_identity"] = {
        "loader_class": type(loader).__name__,
        "root_dir": (
            loader_cfg.get("root_dir")
            if isinstance(loader_cfg, dict)
            else None
        ),
        "preprocessing": effective_preprocessing,
        "settings": loader_settings,
    }

    return complete_config

def _get_verification_pair_statistics(
    labels_pair,
    target_far=0.001,
):
    """
    Build pair-count statistics and warn when the requested FAR is below
    the empirical resolution of the available impostor comparisons.
    """
    pair_statistics = _summarize_verification_pairs(
        labels_pair,
        target_far=target_far,
    )

    if not pair_statistics[
        "Target FAR Empirically Resolvable"
    ]:
        impostor_count = pair_statistics["Impostor Pairs"]
        minimum_far = pair_statistics[
            "Minimum Non-Zero Empirical FAR"
        ]

        if minimum_far is None:
            print(
                "[WARN] Verification evaluation contains no "
                "impostor comparisons."
            )
        else:
            print(
                "[WARN] TAR@0.1%FAR is below the empirical FAR "
                f"resolution supported by {impostor_count} impostor "
                f"comparisons. Minimum non-zero FAR={minimum_far:.6f}."
            )

    return pair_statistics

# =============================================================================
# TASK 1: CLOSED-SET IDENTIFICATION
# =============================================================================
def run_closed_set_identification(x, y, model_class, epochs=150, batch_size=256, 
                                  lr=1e-3, test_split=0.2, val_split=0.0, seed=42, 
                                  device=None, visualize=False, use_template=False, 
                                  template_fusion_method='mean', template_size=None,
                                  matching_method='cosine', outlier_filtering_on_train=False,
                                  outlier_filtering_on_test=False, sqi_scores=None,
                                  sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3,
                                  save_results_and_settings=False, loader=None, 
                                  n_runs=1, _return_stats=False,
                                  intelligent_weight_loading=True):
    """
    Standard Closed-Set Identification Pipeline (Intra-session).
    Determines "Who is this person?" from a known pool of subjects seen during training.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of the data to hold out for testing (0.0 to 1.0).
        val_split (float): Fraction of the training data to use for early stopping validation.
        seed (int): Random seed for reproducibility across splits and weights.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, displays and optionally saves a Confusion Matrix.
        use_template (bool): 
            - False: Uses standard end-to-end Softmax classification.
            - True: Strips the Softmax layer and uses the network as a feature extractor. 
                    Matches Test probes against Train templates.
        template_fusion_method (str): Logic used to create the subject templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of beats used to form the template. None uses all available.
        matching_method (str): Distance/Similarity metric for template matching.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the Training set.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the Test set.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive probe beats to average before making a decision.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split': test_split, 'val_split': val_split, 'use_template': use_template,
        'template_fusion_method': template_fusion_method, 'template_size': template_size,
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    # 2. MULTI-RUN AGGREGATOR (Handles statistical validation)
    if n_runs > 1:
        # Capture current arguments to repeat the experiment
        call_args = _prepare_multi_run_arguments(locals())
        
        # CLEANUP: Crucial step. Remove internal variables from the dict so we don't 
        # pass 'data_stats' or 'hyperparams' as arguments to the next call.
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: 
            call_args.pop(k, None)
        
        # Configure the sub-runs: 
        # - Disable their internal logging (we log the aggregate instead)
        # - Tell them to return their stats so we can capture them
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs) for Statistical Validation...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False # Prevent 5 pop-up windows
            
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            
            # Recursive call to execute a single seed
            res, d_stats, h_params = run_closed_set_identification(**call_args) 
            
            results.append(res)
            # Preserve metadata from the last successful run for the final log file
            data_stats = d_stats  
            hyperparams = h_params 
                
        # Aggregate metrics across all runs
        r1_vals = [r[0] for r in results]
        r5_vals = [r[1] for r in results]
        r1_mean, r1_std = np.mean(r1_vals), np.std(r1_vals)
        r5_mean, r5_std = np.mean(r5_vals), np.std(r5_vals)
        
        print(f"\n[MULTI-RUN RESULTS | {n_runs} Runs]")
        print(f"Rank-1 Acc: {r1_mean:.4f} ± {r1_std:.4f} | Rank-5 Acc: {r5_mean:.4f} ± {r5_std:.4f}")
        
        if save_results_and_settings:
            # Update log metadata with run count and formatted mean/std strings
            hyperparams['n_runs'] = n_runs
            metrics_dict = {
                "Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", 
                "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"
            }
            _log_experiment_results("Closed-Set Identification", metrics_dict, data_stats, hyperparams, loader)
        
        # Return the aggregated statistics
        return (r1_mean, r1_std), (r5_mean, r5_std)

    _set_seed(seed); device = _get_device(device)
    task_title = "Closed-Set Identification"
    mode_str = f"Template ({template_fusion_method}, {matching_method})" if use_template else "Softmax"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Device: {device}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            # Explicitly turn off the flags so the rest of the pipeline safely ignores filtering
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP (Ensures stratify doesn't crash)
    # ====================================================
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 2] # Need at least 2 beats to split Train/Test
    valid_mask = np.isin(y, valid_classes)
    
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    if sqi_scores is not None:
        X_train, X_test, y_train, y_test, sqi_train, sqi_test = train_test_split(
            x, y, sqi_scores, test_size=test_split, stratify=y, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split, stratify=y, random_state=seed
        )
        sqi_train, sqi_test = None, None

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        X_train, y_train = _apply_outlier_filter(
            X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct
        )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test, y_test, sqi_test, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, apply_subject_ranking=False
        )

    # ====================================================
    # 5. CLASS SYNCHRONIZATION & ENCODING
    # ====================================================
    valid_train_classes = np.unique(y_train)
    original_classes = np.unique(y) # 'y' holds the original classes before the split/filter
    
    # Find classes that existed before filtering but are now completely gone
    dropped_classes = np.setdiff1d(original_classes, valid_train_classes)
    
    if len(dropped_classes) > 0:
        print(f"[WARN] Aggressive filtering completely removed {len(dropped_classes)} subjects: {dropped_classes.tolist()}")
    
    # If the filter deleted a class entirely from Train, we MUST drop it from Test
    test_mask = np.isin(y_test, valid_train_classes)
    orphaned_test_beats = len(y_test) - np.sum(test_mask)
    
    if orphaned_test_beats > 0:
        print(f"[WARN] Dropping {orphaned_test_beats} Test beats because their corresponding Train subjects were filtered out.")

    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(valid_train_classes) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough subjects left to continue!")

    # Encode labels safely based ONLY on surviving Train classes
    y_train_enc, classes = _encode_labels(y_train)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test])

    # ====================================================
    # 6. RESUME STANDARD PIPELINE
    # ====================================================
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Data Split: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test)}")
    else:
        X_tr, y_tr = X_train, y_train_enc
        X_val, val_loader = None, None
        print(f"Data Split: Train={len(X_tr)}, Val=0, Test={len(X_test)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    test_loader = _make_loader(X_test, y_test_enc, batch_size, shuffle=False)
 
    # 3. Train (Always start with Softmax training)
    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
        "training_regime": "intra_session_closed_set",
        "model": model_class.__name__,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "val_split": val_split,
        "seed": seed,
        "outlier_train": outlier_filtering_on_train,
        "sqi_thresh": sqi_threshold,
        "classes": len(classes),
        "data_shape": X_tr.shape 
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)

    # 4. Evaluation Logic
    if not use_template:
        # --- PATH A: Standard Softmax ---
        model.eval()
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                probs = torch.softmax(model(xb), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        final_scores = np.vstack(all_probs)
        final_labels = np.concatenate(all_trues)
        
    else:
        # --- PATH B: Template Matching ---
        print(f"[INFO] Building Templates using '{template_fusion_method}' (Beats: {template_size or 'All'})...")
        
        # Switch to Feature Extractor
        model.include_top = False 
        
        # Extract Embeddings from TRAIN set (Enrollment)
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        
        # Create Templates
        templates, temp_labels = _create_templates(
            train_emb, train_lab, method=template_fusion_method, max_beats=template_size
        )
        
        # Extract Embeddings from TEST set (Probe)
        test_emb, test_lab = _get_embeddings(model, test_loader, device)
        
        # Matching
        raw_scores = _compute_score_matrix(test_emb, templates, method=matching_method)
            
        # Required if template_fusion_method='none' leaves multiple templates per subject
        scores = np.full((len(test_emb), len(classes)), -np.inf)
        for class_idx in range(len(classes)):
            gallery_mask = (temp_labels == class_idx)
            if np.any(gallery_mask):
                scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)
                
        final_scores = scores
        final_labels = test_lab
        
        # Restore model
        model.include_top = True

    # ====================================================
    # 5. APPLY SCORE-LEVEL FUSION (If requested)
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        final_scores, final_labels, fusion_size=probe_fusion_size
    )

    if visualize:
        viz = Visualizer()
        preds = np.argmax(final_scores, axis=1)
        viz.plot_confusion_matrix(final_labels, preds, normalize=True)

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Subjects": len(classes),
        "Train Samples": len(X_tr),
        "Validation Samples": len(X_val) if X_val is not None else 0,
        "Test (Probe) Samples": len(X_test),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
        )

    return rank1, rank5

# =============================================================================
# TASK 2: CLOSED-SET VERIFICATION
# =============================================================================
def run_closed_set_verification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3,
                                test_split=0.2, val_split=0.0, num_pairs=10000, 
                                sampling_mode="all", seed=42, device=None, visualize=False,
                                use_template=False, template_fusion_method='mean', 
                                template_size=None, matching_method='cosine',
                                outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                use_deployment_evaluation=False, save_results_and_settings=False, 
                                loader=None, n_runs=1, _return_stats=False,
                                intelligent_weight_loading=True):
    """
    Standard Closed-Set Verification Pipeline (Intra-session).
    Determines "Is this person who they claim to be?" (1:1 matching) for subjects known to the model.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of the data to hold out for testing (0.0 to 1.0).
        val_split (float): Fraction of the training data to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate for evaluation.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the test embeddings.
        use_template (bool): 
            - False: Evaluates raw feature space (Unseen test beats paired vs other Unseen test beats).
            - True: Simulates real-world authentication (Test probes paired against Train templates).
        template_fusion_method (str): Logic used to create the subject templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of beats used to form the template. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the Training set.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the Test set.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): If True, calculates a Global Optimal Threshold on the Validation 
                                          set first, and applies it to the Test set to simulate deployment.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Closed-Set Verification",
    )

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split': test_split, 'val_split': val_split, 'num_pairs': num_pairs,
        'sampling_mode': sampling_mode, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size,
        'matching_method': matching_method,
        'outlier_filter_train': outlier_filtering_on_train, 
        'outlier_filter_test': outlier_filtering_on_test,
        'use_deployment_eval': use_deployment_evaluation
    }

    # --- MULTI-RUN AGGREGATOR ---
    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: 
            call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_closed_set_verification(**call_args) 
            results.append(res)
            data_stats = d_stats
            hyperparams = h_params
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results("Closed-Set Verification", metrics_dict, data_stats, hyperparams, loader)
        return tuple(zip(means, stds))
    # ----------------------------

    _set_seed(seed); device = _get_device(device)
    task_title = "Closed-Set Verification"
    mode_str = f"Template ({template_fusion_method}, size={template_size})" if use_template else "Cloud Pairs (Test Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP (Ensures stratify doesn't crash)
    # ====================================================
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 2] # Need at least 2 beats to split Train/Test
    valid_mask = np.isin(y, valid_classes)
    
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT DATA & SQI SCORES
    # ====================================================
    if sqi_scores is not None:
        X_train, X_test, y_train, y_test, sqi_train, sqi_test = train_test_split(
            x, y, sqi_scores, test_size=test_split, stratify=y, random_state=seed
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_split, stratify=y, random_state=seed
        )
        sqi_train, sqi_test = None, None

    # ====================================================
    # 4. APPLY FILTERS INDEPENDENTLY
    # ====================================================
    if outlier_filtering_on_train and sqi_train is not None:
        print("\n[INFO] Filtering Train Set (Enrollment)...")
        X_train, y_train = _apply_outlier_filter(
            X_train, y_train, sqi_train, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct
        )

    if outlier_filtering_on_test and sqi_test is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test, y_test, sqi_test, absolute_threshold=sqi_threshold, keep_percentage=sqi_keep_pct, apply_subject_ranking=False
        )

    # ====================================================
    # 5. CLASS SYNCHRONIZATION & ENCODING
    # ====================================================
    valid_train_classes = np.unique(y_train)
    original_classes = np.unique(y)
    
    dropped_classes = np.setdiff1d(original_classes, valid_train_classes)
    if len(dropped_classes) > 0:
        print(f"[WARN] Aggressive filtering completely removed {len(dropped_classes)} subjects: {dropped_classes.tolist()}")
    
    test_mask = np.isin(y_test, valid_train_classes)
    orphaned_test_beats = len(y_test) - np.sum(test_mask)
    if orphaned_test_beats > 0:
        print(f"[WARN] Dropping {orphaned_test_beats} Test beats because their corresponding Train subjects were filtered out.")

    X_test, y_test = X_test[test_mask], y_test[test_mask]

    if len(valid_train_classes) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough subjects left to continue!")

    y_train_enc, classes = _encode_labels(y_train)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test])

    # ====================================================
    # 6. RESUME STANDARD PIPELINE
    # ====================================================
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Data Split: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test)}")
    else:
        X_tr, y_tr = X_train, y_train_enc
        X_val, val_loader = None, None
        print(f"Data Split: Train={len(X_tr)}, Val=0, Test={len(X_test)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    test_loader = _make_loader(X_test, y_test_enc, batch_size, shuffle=False)
    
    # 7. Train Model
    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "intra_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
        
    # Switch model to feature extractor
    model.include_top = False

    # ====================================================
    # 8. MODEL CALIBRATION (Optional)
    # ====================================================
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader
        calib_name = "Validation"
        
        print(f"[INFO] Extracting features for Calibration ({calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
    
    # Extract Test Embeddings (Probes)
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # ====================================================
    # 9. EVALUATION STRATEGY
    # ====================================================
    if not use_template:
        # STRATEGY A: Test vs Test (Intra-session unseen evaluation)
        print(f"[INFO] Bypassing Templates. Generating pairs exclusively from Test split...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, 
            labels1=test_lab, 
            embeddings2=None, # None forces Test vs Test matching
            labels2=None, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
    else:
        # STRATEGY B: Test vs Train Templates (Authentication Simulation)
        print(f"[INFO] Building Enrollment Templates from Train split...")
        # [FIX]: Use y_train_enc to match the neural network encoding output
        train_extract_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=False)
        train_emb, train_lab = _get_embeddings(model, train_extract_loader, device)
        
        templates, temp_labels = _create_templates(
            train_emb, train_lab, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, # Probes
            labels1=test_lab, 
            embeddings2=templates, # Enrollment
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Verification Test Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    # ====================================================
    # 10. SAVE RESULTS
    # ====================================================
    data_stats = {
        "Total Subjects": len(classes),
        "Train Samples": len(X_tr),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Test (Probe) Samples": len(X_test),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
        )

    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 3: SUBJECT-DISJOINT IDENTIFICATION (TEMPLATE MATCHING)
# =============================================================================
def run_subject_disjoint_identification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3, 
                                        test_split=0.2, val_split=0.0, seed=42, device=None, 
                                        visualize=False, use_template=True, 
                                        template_fusion_method='mean', template_size=1, 
                                        matching_method='cosine', outlier_filtering_on_train=False, 
                                        outlier_filtering_on_test=False, sqi_scores=None, 
                                        sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3, 
                                        save_results_and_settings=False, loader=None, 
                                        n_runs=1, _return_stats=False,
                                        intelligent_weight_loading=True):
    """
    Subject-Disjoint Identification Pipeline.
    Evaluates identification performance on subjects entirely UNSEEN during the training phase.
    The model learns generalized feature representations on Subject Group A, and builds a gallery for Subject Group B.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to hold out for the disjoint test set.
        val_split (float): Fraction of training subjects to use for early stopping.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen embeddings.
        use_template (bool): MUST be True for Subject-Disjoint identification (requires a gallery).
        template_fusion_method (str): Logic used to enroll unseen subjects into the gallery.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int): Number of chronological beats (e.g., first 5) used to form the gallery template.
        matching_method (str): Distance/Similarity metric used to search the gallery.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the representation learning phase.
        outlier_filtering_on_test (bool): If True, filters noisy beats before forming gallery/probes.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive probe beats to average before searching the gallery.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """   

    # --- ENFORCE OUR AGREED TERMINOLOGY ---
    if not use_template:
        raise ValueError(
            "[ERROR] use_template=False is invalid for Identification. "
            "Identification is a 1:N search and MUST have a defined gallery/reference set. "
            "Please set use_template=True. (If you want to evaluate raw beats without averaging, "
            "set use_template=True and template_fusion_method='none')."
        )
    # --------------------------------------
    
    if template_size is None:
        template_size = 1 # Fallback to single-shot enrollment

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split_subjects': test_split, 'val_split': val_split,
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_subject_disjoint_identification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results("Subject-Disjoint Identification", metrics_dict, data_stats, hyperparams, loader)
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    _set_seed(seed); device = _get_device(device)
    task_title = "Subject-Disjoint Identification"
    mode_str = f"Gallery: First {template_size} beats | Fusion: {template_fusion_method}"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP 
    # ====================================================
    # Test subjects absolutely MUST have enough beats for Gallery + at least 1 Probe
    min_required = template_size + 1 
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_required]
    
    valid_mask = np.isin(y, valid_classes)
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT SUBJECTS (Strictly Disjoint)
    # ====================================================
    y_enc, classes = _encode_labels(y)
    unique_subjs = np.unique(y_enc)
    
    train_subs_full, test_subs = train_test_split(unique_subjs, test_size=test_split, random_state=seed)
    
    if val_split > 0.0:
        train_subs, val_subs = train_test_split(train_subs_full, test_size=val_split, random_state=seed)
        val_mask = np.isin(y_enc, val_subs)
        X_val, y_val = x[val_mask], y_enc[val_mask]
        if sqi_scores is not None: sqi_val = sqi_scores[val_mask]
        print(f"Subject Split: Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
    else:
        train_subs = train_subs_full
        X_val, y_val, sqi_val, val_subs = None, None, None, None
        print(f"Subject Split: Train={len(train_subs)}, Val=0, Test={len(test_subs)}")

    train_mask = np.isin(y_enc, train_subs)
    X_train, y_train = x[train_mask], y_enc[train_mask]
    if sqi_scores is not None: sqi_train = sqi_scores[train_mask]
    
    test_mask = np.isin(y_enc, test_subs)
    X_test, y_test = x[test_mask], y_enc[test_mask]
    if sqi_scores is not None: sqi_test = sqi_scores[test_mask]

    # ====================================================
    # 4. APPLY SQI FILTERS
    # ====================================================
    if outlier_filtering_on_train and sqi_scores is not None:
        print("\n[INFO] Filtering Train Set (Representation Learning)...")
        X_train, y_train = _apply_outlier_filter(X_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)
        # Note: We usually DO NOT filter the Val set to keep early stopping realistic.

    if outlier_filtering_on_test and sqi_scores is not None:
        print("\n[INFO] Filtering Test Set (Gallery & Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 5. POST-FILTER SYNCHRONIZATION
    # ====================================================
    # Ensure Test subjects still have enough beats AFTER filtering
    test_subjs_surviving, test_counts = np.unique(y_test, return_counts=True)
    valid_test_subs = test_subjs_surviving[test_counts >= min_required]
    
    dropped_test_subs = len(test_subs) - len(valid_test_subs)
    if dropped_test_subs > 0:
        print(f"[WARN] Dropping {dropped_test_subs} Test subjects who lost too many beats during filtering to form a Gallery+Probe.")
        
    test_survivor_mask = np.isin(y_test, valid_test_subs)
    X_test, y_test = X_test[test_survivor_mask], y_test[test_survivor_mask]
    test_subs_final = valid_test_subs
    
    if len(test_subs_final) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough Test subjects left to evaluate!")

    # Remap Train Labels to 0..N-1 so CrossEntropy is happy
    y_train_remap, train_classes = _encode_labels(y_train)
    num_train_classes = len(train_classes)
    
    if num_train_classes < 2:
        raise ValueError("[ERROR] Too few Train subjects remaining after filtering.")

    # ====================================================
    # 6. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=seed
        )
        # Create loader ONLY if we actually made a split
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_remap
        # Gracefully assign None without calling _make_loader
        val_loader_seen = None
    
    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    
    # This remains the UNSEEN Validation subjects loader
    val_loader_unseen = _make_loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None
    
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    
    model = model_class(in_channels=_detect_channels(x), num_classes=num_train_classes, include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "intra_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Subject-Disjoint model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            cache.save_weight_cache(model, train_config, uid)
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Call the updated custom loop passing both validation loaders of seen and unseen subjects!
        model = _run_train_loop_unseen_subjects(
            model, train_loader, val_loader_seen, val_loader_unseen, optimizer, criterion, device, 
            epochs, matching_method=matching_method, patience=40, lr_patience=15
        )

    # ====================================================
    # 7. FINAL INFERENCE ON UNSEEN TEST SUBJECTS
    # ====================================================
    model.include_top = False
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    print(f"[INFO] Splitting Test Data: First {template_size} beats = Gallery, Rest = Probe")
    enroll_embs_list, enroll_y_list = [], []
    probe_embs_list, probe_y_list = [], []
    
    # Map disjoint test subject IDs to 0..N_test-1 for the identification metric array
    test_sub_map = {sub: i for i, sub in enumerate(test_subs_final)}
    
    for sub in test_subs_final:
        sub_idxs = np.where(test_lab == sub)[0]
        # We already guaranteed sub_idxs > template_size in Step 5!
        enroll_idx = sub_idxs[:template_size]
        probe_idx = sub_idxs[template_size:]
        
        enroll_embs_list.append(test_emb[enroll_idx])
        enroll_y_list.append(test_lab[enroll_idx])
        probe_embs_list.append(test_emb[probe_idx])
        probe_y_list.append(test_lab[probe_idx])

    emb_enroll = np.vstack(enroll_embs_list)
    lab_enroll = np.concatenate(enroll_y_list)
    emb_probe = np.vstack(probe_embs_list)
    lab_probe = np.concatenate(probe_y_list)
    
    # 8. Apply Template Fusion Strategy to the Gallery
    gallery_emb, gallery_lab = _create_templates(
        emb_enroll, lab_enroll, method=template_fusion_method, max_beats=None
    )
    
    # 9. Generate Score Matrix for Rank-N Evaluation
    probe_mapped = np.array([test_sub_map[l] for l in lab_probe])
    
    raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
    scores = np.full((len(emb_probe), len(test_subs_final)), -np.inf)
    
    for class_idx, sub in enumerate(test_subs_final):
        gallery_mask = (gallery_lab == sub)
        if np.any(gallery_mask):
            scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)

    # ====================================================
    # 10. APPLY SCORE-LEVEL FUSION
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        scores, probe_mapped, fusion_size=probe_fusion_size
    )

    if visualize:
        # Visualizing original un-fused test embeddings
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Unseen Subject Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    data_stats = {
        "Train Subjects": len(train_subs),
        "Train Samples": len(X_train),
        "Validation Subjects": (
            len(val_subs)
            if val_subs is not None
            else 0
        ),
        "Validation Samples": (
            len(X_val)
            if X_val is not None
            else 0
        ),
        "Test Subjects": len(test_subs_final),
        "Gallery Size": len(gallery_emb),
        "Probe Samples": len(emb_probe),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
        )

    # 11. Report Identification Metrics
    return rank1, rank5

# =============================================================================
# TASK 4: SUBJECT-DISJOINT VERIFICATION
# =============================================================================
def run_subject_disjoint_verification(x, y, model_class, epochs=150, batch_size=256, lr=1e-3, 
                                      test_split=0.2, val_split=0.0, num_pairs=10000, 
                                      sampling_mode="all", seed=42, device=None, 
                                      visualize=False, use_template=False, template_fusion_method='mean', 
                                      template_size=1, matching_method='cosine',
                                      outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                      sqi_scores=None, sqi_threshold=0.05, sqi_keep_pct=0.8, 
                                      use_deployment_evaluation=False, save_results_and_settings=False, 
                                      loader=None, n_runs=1, _return_stats=False,
                                      intelligent_weight_loading=True):
    """
    Subject-Disjoint Verification Pipeline (Subject-Disjoint 1:1 Matching).
    Tests the system's ability to verify the identity of completely new users.
    The model is trained on Subject Group A and evaluated via pairs constructed from Subject Group B.

    Args:
        x (np.ndarray): Input ECG signals or features.
        y (np.ndarray): Subject class labels corresponding to x.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to hold out for the disjoint test set.
        val_split (float): Fraction of training subjects to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen embeddings.
        use_template (bool): 
            - False: "Cloud-based" matching (Random pairs formed entirely within the unseen Test group).
            - True: "Authentication" simulation (Unseen subjects' later beats matched against their initial beats).
        template_fusion_method (str): Logic used to create templates if use_template is True.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int): Number of initial beats to form the enrollment template for unseen subjects.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): If True, filters noisy beats from the representation learning phase.
        outlier_filtering_on_test (bool): If True, filters noisy beats from the verification test subjects.
        sqi_scores (str or np.ndarray): SQI calculation method (e.g., 'kurtosis') or pre-computed array.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses an unseen validation group to find a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Verification",
    )

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 
        'test_split_subjects': test_split, 'val_split': val_split, 'num_pairs': num_pairs,
        'use_template': use_template, 'template_fusion_method': template_fusion_method, 
        'template_size': template_size, 'matching_method': matching_method, 
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_subject_disjoint_verification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results("Subject-Disjoint Verification", metrics_dict, data_stats, hyperparams, loader)
        return tuple(zip(means, stds))
    # ----------------------------

    if use_template and template_size is None:
        template_size = 1

    _set_seed(seed); device = _get_device(device)       
    task_title = "Subject-Disjoint Verification"        
    mode_str = f"Template ({template_fusion_method}, First {template_size})" if use_template else "Cloud Pairs (Test Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION
    # ====================================================
    if outlier_filtering_on_train or outlier_filtering_on_test:
        if sqi_scores is None:
            print("[WARN] Filtering requested but sqi_scores is None. Skipping filtering entirely.")
            outlier_filtering_on_train = False
            outlier_filtering_on_test = False
        elif isinstance(sqi_scores, str):
            print(f"[INFO] Calculating SQI using method: '{sqi_scores}'")
            sqi_scores = np.array(_compute_sqi(x, method=sqi_scores))
        elif isinstance(sqi_scores, (list, np.ndarray)):
            sqi_scores = np.array(sqi_scores)
        else:
            raise TypeError("[ERROR] sqi_scores must be a string, array, or None.")
    else:
        sqi_scores = None

    # ====================================================
    # 2. PRE-SPLIT CLEANUP 
    # ====================================================
    # If using templates, subjects need enough beats for Gallery + Probes.
    # If not using templates, they just need at least 2 beats to form pairs.
    min_required = (template_size + 1) if use_template else 2
    
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_required]
    
    valid_mask = np.isin(y, valid_classes)
    x, y = x[valid_mask], y[valid_mask]
    if sqi_scores is not None: 
        sqi_scores = sqi_scores[valid_mask]

    # ====================================================
    # 3. SPLIT SUBJECTS (Strictly Disjoint)
    # ====================================================
    y_enc, classes = _encode_labels(y)
    unique_subjs = np.unique(y_enc)
    
    train_subs_full, test_subs = train_test_split(unique_subjs, test_size=test_split, random_state=seed)
    
    if val_split > 0.0:
        train_subs, val_subs = train_test_split(train_subs_full, test_size=val_split, random_state=seed)
        val_mask = np.isin(y_enc, val_subs)
        X_val, y_val = x[val_mask], y_enc[val_mask]
        if sqi_scores is not None: sqi_val = sqi_scores[val_mask]
        print(f"Subject Split: Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
    else:
        train_subs = train_subs_full
        X_val, y_val, sqi_val, val_subs = None, None, None, None
        print(f"Subject Split: Train={len(train_subs)}, Val=0, Test={len(test_subs)}")

    train_mask = np.isin(y_enc, train_subs)
    X_train, y_train = x[train_mask], y_enc[train_mask]
    if sqi_scores is not None: sqi_train = sqi_scores[train_mask]
    
    test_mask = np.isin(y_enc, test_subs)
    X_test, y_test = x[test_mask], y_enc[test_mask]
    if sqi_scores is not None: sqi_test = sqi_scores[test_mask]

    # ====================================================
    # 4. APPLY SQI FILTERS
    # ====================================================
    if outlier_filtering_on_train and sqi_scores is not None:
        print("\n[INFO] Filtering Train Set (Representation Learning)...")
        X_train, y_train = _apply_outlier_filter(X_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if outlier_filtering_on_test and sqi_scores is not None:
        print("\n[INFO] Filtering Test Set (Probes)...")
        X_test, y_test = _apply_outlier_filter(
            X_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 5. POST-FILTER SYNCHRONIZATION
    # ====================================================
    test_subjs_surviving, test_counts = np.unique(y_test, return_counts=True)
    valid_test_subs = test_subjs_surviving[test_counts >= min_required]
    
    dropped_test_subs = len(test_subs) - len(valid_test_subs)
    if dropped_test_subs > 0:
        print(f"[WARN] Dropping {dropped_test_subs} Test subjects who lost too many beats during filtering.")
        
    test_survivor_mask = np.isin(y_test, valid_test_subs)
    X_test, y_test = X_test[test_survivor_mask], y_test[test_survivor_mask]
    test_subs_final = valid_test_subs
    
    if len(test_subs_final) < 2:
        raise ValueError("[ERROR] Data filtering was too aggressive. Not enough Test subjects left to evaluate!")

    y_train_remap, train_classes = _encode_labels(y_train)
    num_train_classes = len(train_classes)
    
    if num_train_classes < 2:
        raise ValueError("[ERROR] Too few Train subjects remaining after filtering.")

# ====================================================
    # 6. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_remap, test_size=val_split, stratify=y_train_remap, random_state=seed
        )
        # Create loader ONLY if we actually made a split
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_remap
        # Gracefully assign None without calling _make_loader
        val_loader_seen = None
    
    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    
    # This remains the UNSEEN Validation subjects loader (used for EER)
    val_loader_unseen = _make_loader(X_val, y_val, batch_size, shuffle=False) if X_val is not None else None
    
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    
    model = model_class(in_channels=_detect_channels(x), num_classes=num_train_classes, include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "intra_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "matching_method": matching_method  # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Subject-Disjoint model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            cache.save_weight_cache(model, train_config, uid)
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 7. MODEL CALIBRATION (Optional)
    # ====================================================
    model.include_top = False
    
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader_unseen
        calib_name = "Unseen Validation"
        
        print(f"[INFO] Extracting features for Calibration ({calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")
        
    # ====================================================
    # 8. FINAL INFERENCE ON UNSEEN TEST SUBJECTS
    # ====================================================
    test_emb, test_lab = _get_embeddings(model, test_loader, device)

    # 9. Evaluation Strategy
    if not use_template:
        print(f"[INFO] Bypassing Templates. Generating pairs entirely from Unseen Test Subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=test_emb, labels1=test_lab, 
            embeddings2=None, labels2=None, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
    else:
        print(f"[INFO] Splitting Test Data: First {template_size} beats = Enroll, Rest = Probe")
        enroll_embs_list, enroll_y_list = [], []
        probe_embs_list, probe_y_list = [], []
        
        # Use test_subs_final to guarantee valid indexing
        for sub in test_subs_final:
            sub_idxs = np.where(test_lab == sub)[0]
            
            enroll_idx = sub_idxs[:template_size]
            probe_idx = sub_idxs[template_size:]
            
            enroll_embs_list.append(test_emb[enroll_idx])
            enroll_y_list.append(test_lab[enroll_idx])
            probe_embs_list.append(test_emb[probe_idx])
            probe_y_list.append(test_lab[probe_idx])
            
        if len(enroll_embs_list) == 0:
            print("[WARN] Not enough beats per subject for this template size.")
            return 0.0, 0.0, 0.0, 0.0

        emb_enroll = np.vstack(enroll_embs_list)
        lab_enroll = np.concatenate(enroll_y_list)
        emb_probe = np.vstack(probe_embs_list)
        lab_probe = np.concatenate(probe_y_list)
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=None
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, labels1=lab_probe, 
            embeddings2=templates, labels2=temp_labels, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(test_emb, test_lab, title="Unseen Subject Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    # Update hyperparams dictionary dynamically using the local 'ep' variable
    # hyperparams['epochs'] = f"{epochs} (stopped at {ep + 1})" if (ep + 1) < epochs else epochs
    
    data_stats = {
        "Train Subjects": len(train_subs),
        "Train Samples": len(X_train),
        "Test Subjects": len(test_subs_final),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
        )

    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 5: CROSS-SESSION IDENTIFICATION
# =============================================================================
def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=150, 
                                     batch_size=256, lr=1e-3, val_split=0.0, seed=42, device=None, 
                                     visualize=False, use_template=False, template_fusion_method='mean',
                                     template_size=None, matching_method='cosine',
                                     outlier_filtering_on_train=False, outlier_filtering_on_test=False, 
                                     sqi_train=None, sqi_test=None, sqi_threshold=0.05, 
                                     sqi_keep_pct=0.8, probe_fusion_size=3, save_results_and_settings=False, 
                                     loader=None, n_runs=1, _return_stats=False,
                                     intelligent_weight_loading=True):
    """
    Cross-Session Identification Pipeline (Temporal Robustness).
    Evaluates system robustness against physiological aging and sensor variations over time.
    Trains the model on Session 1 (Enrollment) and identifies subjects using Session 2 (Probes).

    Args:
        x_train (np.ndarray): Input ECG signals from Session 1.
        y_train (np.ndarray): Labels for Session 1.
        x_test (np.ndarray): Input ECG signals from Session 2.
        y_test (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool): 
            - False: Uses the Session 1 Softmax classification weights to classify Session 2 data.
            - True: Uses Session 1 features to form a gallery, and metric-matches Session 2 probes.
        template_fusion_method (str): Logic used to create the Session 1 gallery template.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric for template matching.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering independently to Session 1.
        outlier_filtering_on_test (bool): Apply SQI filtering independently to Session 2.
        sqi_train (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_test (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive Session 2 beats to average before making a decision.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """

    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size, 'val_split': val_split,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_cross_session_identification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results("Cross-Session Identification", metrics_dict, data_stats, hyperparams, loader)
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    _set_seed(seed); device = _get_device(device)
    task_title = "Cross-Session Identification"
    mode_str = f"Template ({template_fusion_method}, size={template_size or 'All'})" if use_template else "Softmax Classifier"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method if use_template else 'N/A'}")
    
    # ====================================================
    # 1. DYNAMIC SQI CALCULATION (Independent Sessions)
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None:
            print(f"[WARN] Filtering requested for {name} but sqi scores are None. Skipping {name} filtering.")
            return None
        if isinstance(sqi_input, str):
            print(f"[INFO] Calculating SQI for {name} using method: '{sqi_input}'")
            return np.array(_compute_sqi(x_data, method=sqi_input))
        if isinstance(sqi_input, (list, np.ndarray)):
            return np.array(sqi_input)
        raise TypeError(f"[ERROR] sqi_{name.lower()} must be a string, array, or None.")

    sqi_train = _prepare_sqi(sqi_train, x_train, outlier_filtering_on_train, "Train")
    sqi_test = _prepare_sqi(sqi_test, x_test, outlier_filtering_on_test, "Test")

    # ====================================================
    # 2. APPLY SQI FILTERS
    # ====================================================
    if sqi_train is not None:
        print("\n[INFO] Filtering Session 1 (Enrollment)...")
        x_train, y_train = _apply_outlier_filter(x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if sqi_test is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_test, y_test = _apply_outlier_filter(
            x_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 3. INTERSECT SUBJECTS (Post-Filter Sync)
    # ====================================================
    train_subs = set(y_train)
    test_subs = set(y_test)
    common_subs = sorted(list(train_subs.intersection(test_subs)))
    
    if len(common_subs) < 2: 
        print("[WARN] Not enough common subjects between sessions after filtering.")
        return 0.0, 0.0
    
    train_mask = np.isin(y_train, common_subs)
    test_mask = np.isin(y_test, common_subs)
    
    x_train_full, y_train_full = x_train[train_mask], y_train[train_mask]
    x_test_filtered, y_test_filtered = x_test[test_mask], y_test[test_mask]

    # Pre-split cleanup: Ensure Train classes have >= 2 beats so stratify doesn't crash
    unique_classes, counts = np.unique(y_train_full, return_counts=True)
    valid_classes = unique_classes[counts >= 2]
    
    if len(valid_classes) < len(common_subs):
        dropped = len(common_subs) - len(valid_classes)
        print(f"[WARN] Dropping {dropped} subjects who have fewer than 2 beats left in Session 1.")
        
    final_train_mask = np.isin(y_train_full, valid_classes)
    final_test_mask = np.isin(y_test_filtered, valid_classes) # Sync Session 2 again!
    
    x_train_full, y_train_full = x_train_full[final_train_mask], y_train_full[final_train_mask]
    x_test_filtered, y_test_filtered = x_test_filtered[final_test_mask], y_test_filtered[final_test_mask]

    # ====================================================
    # 4. ENCODE LABELS
    # ====================================================
    # Encode Labels to 0..N-1 based strictly on the surviving training set classes
    y_train_enc, classes = _encode_labels(y_train_full)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test_filtered])
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    # Validation Split (from Session 1)
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Session 1 Split: Train={len(X_tr)}, Val={len(X_val)} | Session 2 Probes={len(x_test_filtered)}")
    else:
        X_tr, y_tr = x_train_full, y_train_enc
        X_val, val_loader = None, None
        print(f"Session 1 Split: Train={len(X_tr)}, Val=0 | Session 2 Probes={len(x_test_filtered)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    probe_loader = _make_loader(x_test_filtered, y_test_enc, batch_size, shuffle=False)
    
    # Train Model
    model = model_class(in_channels=_detect_channels(x_train_full), num_classes=len(classes), include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "cross_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    
    # ====================================================
    # 6. EVALUATION STRATEGY
    # ====================================================
    if not use_template:
        # STRATEGY A: Standard Softmax Classification
        print("[INFO] Bypassing Templates. Using standard Softmax Classifier trained on Session 1...")
        model.eval()
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in probe_loader:
                xb = xb.to(device)
                probs = torch.softmax(model(xb), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_trues.append(yb.cpu().numpy())
        final_scores = np.vstack(all_probs)
        final_labels = np.concatenate(all_trues)
        
    else:
        # STRATEGY B: Template Matching
        print(f"[INFO] Building Enrollment Templates from Session 1...")
        model.include_top = False # Switch to Feature Extractor
        
        enroll_loader = _make_loader(x_train_full, y_train_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        gallery_emb, gallery_lab = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

        # raw_scores shape: (N_Probes, N_Gallery_Items)
        raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
        
        # Collapse raw_scores into class scores cleanly
        scores = np.full((len(emb_probe), len(classes)), -np.inf)
        for class_idx in range(len(classes)):
            gallery_mask = (gallery_lab == class_idx)
            if np.any(gallery_mask):
                scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)
                
        final_scores = scores
        final_labels = lab_probe
        
        # Restore model
        model.include_top = True

    # ====================================================
    # 7. APPLY SCORE-LEVEL FUSION
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(
        final_scores, final_labels, fusion_size=probe_fusion_size
    )

    if visualize and use_template:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, final_labels, title="Cross-Session Probe Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Cross-Session Subjects": len(classes),
        "Enrollment (S1) Samples": len(x_train_full),
        "Probe (S2) Samples": len(x_test_filtered),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
        )

    # 8. Report Identification Metrics
    return rank1, rank5

# =============================================================================
# TASK 6: CROSS-SESSION VERIFICATION
# =============================================================================
def run_cross_session_verification(x_train, y_train, x_test, y_test, model_class, epochs=150, 
                                   batch_size=256, lr=1e-3, val_split=0.0, num_pairs=10000, 
                                   sampling_mode="all", seed=42, device=None, visualize=False, 
                                   use_template=False, template_fusion_method='mean', template_size=None, 
                                   matching_method='cosine', outlier_filtering_on_train=False, 
                                   outlier_filtering_on_test=False, sqi_train=None, sqi_test=None, 
                                   sqi_threshold=0.05, sqi_keep_pct=0.8, use_deployment_evaluation=False, 
                                   save_results_and_settings=False, loader=None, 
                                   n_runs=1, _return_stats=False,
                                   intelligent_weight_loading=True):
    """
    Cross-Session Verification Pipeline (Temporal Robustness 1:1).
    Attempts to verify if a subject is who they claim to be across different time-separated recording sessions.

    Args:
        x_train (np.ndarray): Input ECG signals from Session 1.
        y_train (np.ndarray): Labels for Session 1.
        x_test (np.ndarray): Input ECG signals from Session 2.
        y_test (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        val_split (float): Fraction of Session 1 data to use for early stopping.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the cross-session embeddings.
        use_template (bool):
            - False: Evaluates raw temporal space (Session 2 beats paired vs other Session 2 beats).
            - True: Simulates Authentication (Session 2 probes paired against Session 1 enrollment templates).
        template_fusion_method (str): Logic used to create the Session 1 templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering independently to Session 1.
        outlier_filtering_on_test (bool): Apply SQI filtering independently to Session 2.
        sqi_train (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_test (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses Session 1 Validation data to calculate a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Cross-Session Verification",
    )
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'num_pairs': num_pairs, 'use_template': use_template, 
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'val_split': val_split,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_cross_session_verification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results("Cross-Session Verification", metrics_dict, data_stats, hyperparams, loader)
        return tuple(zip(means, stds))
    # ----------------------------

    _set_seed(seed); device = _get_device(device)
    task_title = "Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, size={template_size or 'All'})" if use_template else "Cloud Pairs (Session 2 Only)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. DYNAMIC SQI CALCULATION (Independent Sessions)
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None:
            print(f"[WARN] Filtering requested for {name} but sqi scores are None. Skipping {name} filtering.")
            return None
        if isinstance(sqi_input, str):
            print(f"[INFO] Calculating SQI for {name} using method: '{sqi_input}'")
            return np.array(_compute_sqi(x_data, method=sqi_input))
        if isinstance(sqi_input, (list, np.ndarray)):
            return np.array(sqi_input)
        raise TypeError(f"[ERROR] sqi_{name.lower()} must be a string, array, or None.")

    sqi_train = _prepare_sqi(sqi_train, x_train, outlier_filtering_on_train, "Train")
    sqi_test = _prepare_sqi(sqi_test, x_test, outlier_filtering_on_test, "Test")

    # ====================================================
    # 2. APPLY SQI FILTERS
    # ====================================================
    if sqi_train is not None:
        print("\n[INFO] Filtering Session 1 (Enrollment)...")
        x_train, y_train = _apply_outlier_filter(x_train, y_train, sqi_train, sqi_threshold, sqi_keep_pct)

    if sqi_test is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_test, y_test = _apply_outlier_filter(
            x_test,
            y_test,
            sqi_test,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 3. INTERSECT SUBJECTS (Post-Filter Sync)
    # ====================================================
    train_subs = set(y_train)
    test_subs = set(y_test)
    common_subs = sorted(list(train_subs.intersection(test_subs)))
    
    if len(common_subs) < 2: 
        print("[WARN] Not enough common subjects between sessions after filtering.")
        return 0.0, 0.0, 0.0, 0.0
    
    train_mask = np.isin(y_train, common_subs)
    test_mask = np.isin(y_test, common_subs)
    
    x_train_full, y_train_full = x_train[train_mask], y_train[train_mask]
    x_test_filtered, y_test_filtered = x_test[test_mask], y_test[test_mask]

    unique_classes, counts = np.unique(y_train_full, return_counts=True)
    valid_classes = unique_classes[counts >= 2]
    
    if len(valid_classes) < len(common_subs):
        dropped = len(common_subs) - len(valid_classes)
        print(f"[WARN] Dropping {dropped} subjects who have fewer than 2 beats left in Session 1.")
        
    final_train_mask = np.isin(y_train_full, valid_classes)
    final_test_mask = np.isin(y_test_filtered, valid_classes) # Sync Session 2 again!
    
    x_train_full, y_train_full = x_train_full[final_train_mask], y_train_full[final_train_mask]
    x_test_filtered, y_test_filtered = x_test_filtered[final_test_mask], y_test_filtered[final_test_mask]

    # ====================================================
    # 4. ENCODE LABELS
    # ====================================================
    y_train_enc, classes = _encode_labels(y_train_full)
    label_map = {c: i for i, c in enumerate(classes)}
    y_test_enc = np.array([label_map[l] for l in y_test_filtered])
    
    # ====================================================
    # 5. RESUME STANDARD PIPELINE
    # ====================================================
    if val_split > 0.0:
        X_tr, X_val, y_tr, y_val = train_test_split(
            x_train_full, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader = _make_loader(X_val, y_val, batch_size, shuffle=False)
        print(f"Session 1 Split: Train={len(X_tr)}, Val={len(X_val)} | Session 2 Probes={len(x_test_filtered)}")
    else:
        X_tr, y_tr = x_train_full, y_train_enc
        X_val, val_loader = None, None
        print(f"Session 1 Split: Train={len(X_tr)}, Val=0 | Session 2 Probes={len(x_test_filtered)}")

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    probe_loader = _make_loader(x_test_filtered, y_test_enc, batch_size, shuffle=False)
    
    # Train Model
    model = model_class(in_channels=_detect_channels(x_train_full), num_classes=len(classes), include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "cross_session_closed_set",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": len(classes), "data_shape": X_tr.shape
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()    
        model = _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    
    # Switch to Feature Extractor
    model.include_top = False

    # ====================================================
    # 6. MODEL CALIBRATION (Optional)
    # ====================================================
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader
        calib_name = "Validation"
        
        print(f"[INFO] Extracting features for Calibration (Session 1 {calib_name} Set)...")
        calib_emb, calib_lab = _get_embeddings(model, calib_loader, device)
        
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb, labels1=calib_lab, embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 7. EVALUATION STRATEGY
    # ====================================================
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    if not use_template:
        # STRATEGY A: Session 2 vs Session 2 (Intra-session unseen evaluation)
        print(f"[INFO] Bypassing Templates. Generating pairs exclusively from Session 2...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, 
            labels1=lab_probe, 
            embeddings2=None, # None forces test vs test matching
            labels2=None, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
    else:
        # STRATEGY B: Session 2 Probes vs Session 1 Templates (Authentication Simulation)
        print(f"[INFO] Building Enrollment Templates from Session 1...")
        enroll_loader = _make_loader(x_train_full, y_train_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, # Session 2 Probes
            labels1=lab_probe, 
            embeddings2=templates, # Session 1 Templates
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )

    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Cross-Session Probe Embeddings (T-SNE)")

    # 8. Apply Deployment Calibration
    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    # Update hyperparams dictionary dynamically
    hyperparams['epochs'] = f"{epochs} (stopped at {model.actual_epochs})" if model.actual_epochs < epochs else epochs

    data_stats = {
        "Total Cross-Session Subjects": len(classes),
        "Enrollment (S1) Samples": len(x_train_full),
        "Probe (S2) Samples": len(x_test_filtered),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
        )

    # 9. Report Verification Metrics
    return eer, auc_val, dprime, tar

# =============================================================================
# TASK 7: SUBJECT-DISJOINT CROSS-SESSION IDENTIFICATION
# =============================================================================
def run_subject_disjoint_cross_session_identification(
        x_s1, y_s1, x_s2, y_s2, model_class, epochs=150, batch_size=256, lr=1e-3, test_split=0.2, val_split=0.0, 
        seed=42, device=None, visualize=False, use_template=True, template_fusion_method='mean', template_size=None, 
        matching_method='cosine', outlier_filtering_on_train=False, outlier_filtering_on_test=False, sqi_s1=None, 
        sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8, probe_fusion_size=3, save_results_and_settings=False, 
        loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True):
    """
    The Ultimate Biometric Test: Subject-Disjoint + Temporal Robustness Identification.
    1. Trains a feature extractor on Session 1 of Subject Group A.
    2. Enrolls Unseen Subject Group B using their Session 1 recordings to build a gallery.
    3. Identifies Subject Group B using their Session 2 recordings as probes.

    Args:
        x_s1 (np.ndarray): Input ECG signals from Session 1.
        y_s1 (np.ndarray): Labels for Session 1.
        x_s2 (np.ndarray): Input ECG signals from Session 2.
        y_s2 (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen temporal embeddings.
        use_template (bool): MUST be True for this task (requires a gallery to identify unseen subjects).
        template_fusion_method (str): Logic used to enroll unseen Session 1 data into the gallery.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats to form the gallery. None uses all available.
        matching_method (str): Distance/Similarity metric used to search the gallery.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering to Session 1 data.
        outlier_filtering_on_test (bool): Apply SQI filtering to Session 2 data.
        sqi_s1 (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_s2 (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        probe_fusion_size (int): Number of consecutive Session 2 beats to average before searching the gallery.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for both metrics.
    """
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'test_split': test_split, 'val_split': val_split,
        'template_fusion_method': template_fusion_method, 'template_size': template_size, 
        'matching_method': matching_method, 'probe_fusion_size': probe_fusion_size,
        'outlier_filter_train': outlier_filtering_on_train, 'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_subject_disjoint_cross_session_identification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        r1_mean, r1_std = np.mean([r[0] for r in results]), np.std([r[0] for r in results])
        r5_mean, r5_std = np.mean([r[1] for r in results]), np.std([r[1] for r in results])
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {"Rank-1 Accuracy": f"{r1_mean:.4f} ± {r1_std:.4f}", "Rank-5 Accuracy": f"{r5_mean:.4f} ± {r5_std:.4f}"}
            _log_experiment_results("Subject-Disjoint Cross-Session ID", metrics_dict, data_stats, hyperparams, loader)
        return (r1_mean, r1_std), (r5_mean, r5_std)
    # ----------------------------

    if not use_template:
        raise ValueError("[ERROR] use_template=False is invalid for Identification. Must use templates to build a gallery.")
        
    _set_seed(seed); device = _get_device(device)
    task_title = "Subject-Disjoint Cross-Session ID"
    mode_str = f"Gallery: Session 1 ({template_fusion_method}, size={template_size or 'All'})"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Session 1")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering Session 1 (Representation & Enrollment)...")
        x_s1, y_s1 = _apply_outlier_filter(x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct)

    if sqi_s2 is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_s2, y_s2 = _apply_outlier_filter(
            x_s2,
            y_s2,
            sqi_s2,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS (STRICTLY DISJOINT)
    # ====================================================
    # We must only evaluate subjects that successfully completed both sessions.
    common_subs = sorted(list(set(y_s1).intersection(set(y_s2))))
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across sessions after filtering.")
        
    # Split the distinct subjects into Train, Val, and Test cohorts
    train_subs_full, test_subs = train_test_split(common_subs, test_size=test_split, random_state=seed)
    
    if val_split > 0.0:
        train_subs, val_subs = train_test_split(train_subs_full, test_size=val_split, random_state=seed)
        print(f"Subject Split: Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
    else:
        train_subs = train_subs_full
        val_subs = []
        print(f"Subject Split: Train={len(train_subs)}, Val=0, Test={len(test_subs)}")

    # Extract respective datasets based on the disjoint subject split
    X_train, Y_train = x_s1[np.isin(y_s1, train_subs)], y_s1[np.isin(y_s1, train_subs)]
    
    # STRICT TEMPORAL ISOLATION: Validation uses ONLY Session 1 data
    X_val_s1, Y_val_s1 = (x_s1[np.isin(y_s1, val_subs)], y_s1[np.isin(y_s1, val_subs)]) if len(val_subs) > 0 else (None, None)
    
    X_enroll, Y_enroll = x_s1[np.isin(y_s1, test_subs)], y_s1[np.isin(y_s1, test_subs)]
    X_probe, Y_probe = x_s2[np.isin(y_s2, test_subs)], y_s2[np.isin(y_s2, test_subs)]

    # ====================================================
    # 3. ENCODE LABELS (CRITICAL FIX FOR PYTORCH TENSORS)
    # ====================================================
    # PyTorch Datasets cannot handle raw strings (like "MLS" or "HPS").
    # We must explicitly map these strings to integers (0 to N-1).
    
    # A. Train Labels
    y_train_enc, train_classes = _encode_labels(Y_train)
    num_train_classes = len(train_classes)
    
    # B. Validation Labels (Ensures S1 map to the exact same integers)
    if len(val_subs) > 0:
        val_map = {sub: i for i, sub in enumerate(val_subs)}
        y_val_s1_enc = np.array([val_map[s] for s in Y_val_s1])
    else:
        y_val_s1_enc = None

    # C. Test Labels (Ensures Enroll and Probe map to the exact same integers)
    test_map = {sub: i for i, sub in enumerate(test_subs)}
    y_enroll_enc = np.array([test_map[s] for s in Y_enroll])
    y_probe_enc = np.array([test_map[s] for s in Y_probe])

    # ====================================================
    # 4. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_enc
        val_loader_seen = None

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader_s1 = _make_loader(X_val_s1, y_val_s1_enc, batch_size, shuffle=False) if X_val_s1 is not None else None
    
    # UNSEEN Validation now strictly passes the Session 1 loader only (Intra-Session check)
    val_loader_unseen = val_loader_s1
    
    model = model_class(in_channels=_detect_channels(x_s1), num_classes=num_train_classes, include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "cross_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "matching_method": matching_method # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )
        
        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Subject-Disjoint Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 5. FINAL INFERENCE ON UNSEEN SUBJECTS
    # ====================================================
    model.include_top = False # Final metric extraction
    
    print(f"[INFO] Building Enrollment Templates for Unseen Subjects from Session 1...")
    enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
    emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
    
    gallery_emb, gallery_lab = _create_templates(
        emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
    )

    print(f"[INFO] Probing with Unseen Subjects from Session 2...")
    probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    # Because we mapped y_enroll_enc and y_probe_enc to integers, 
    # gallery_lab and lab_probe are already perfectly aligned from 0 to N-1!
    raw_scores = _compute_score_matrix(emb_probe, gallery_emb, method=matching_method)
    scores = np.full((len(emb_probe), len(test_subs)), -np.inf)
    
    for class_idx in range(len(test_subs)):
        gallery_mask = (gallery_lab == class_idx)
        if np.any(gallery_mask):
            scores[:, class_idx] = np.max(raw_scores[:, gallery_mask], axis=1)

    # ====================================================
    # 6. APPLY SCORE-LEVEL FUSION & EVALUATE
    # ====================================================
    final_scores, final_labels = _apply_score_fusion(scores, lab_probe, fusion_size=probe_fusion_size)

    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Disjoint Cross-Session Embeddings (T-SNE)")

    rank1, rank5 = _compute_metrics_identification(final_scores, final_labels)

    # Update hyperparams dictionary dynamically using the model's tracked epochs
    actual_ep = getattr(model, 'actual_epochs', epochs)
    hyperparams['epochs'] = f"{epochs} (stopped at {actual_ep})" if actual_ep < epochs else epochs

    data_stats = {
        "Train Subjects": len(train_subs),
        "Test Subjects": len(test_subs),
        "Train (S1) Samples": len(X_train),
        "Enrollment (S1) Samples": len(X_enroll),
        "Probe (S2) Samples": len(X_probe),
    }

    if _return_stats:
        return (rank1, rank5), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "Rank-1 Accuracy": rank1,
                "Rank-5 Accuracy": rank5,
            },
            data_stats,
            hyperparams,
            loader,
        )

    return rank1, rank5


# =============================================================================
# TASK 8: SUBJECT-DISJOINT CROSS-SESSION VERIFICATION
# =============================================================================
def run_subject_disjoint_cross_session_verification(
        x_s1, y_s1, x_s2, y_s2, model_class, epochs=150, batch_size=256, lr=1e-3, test_split=0.2, val_split=0.0, 
        num_pairs=10000, sampling_mode="all", seed=42, device=None, visualize=False, use_template=False, 
        template_fusion_method='mean', template_size=None, matching_method='cosine', outlier_filtering_on_train=False, 
        outlier_filtering_on_test=False, sqi_s1=None, sqi_s2=None, sqi_threshold=0.05, sqi_keep_pct=0.8,
        use_deployment_evaluation=False, save_results_and_settings=False, loader=None, n_runs=1, _return_stats=False,
        intelligent_weight_loading=True):
    """
    The Ultimate Biometric Test: Subject-Disjoint + Temporal Robustness 1:1 Verification.
    Verifies the identity of subjects completely excluded from representation learning, across different recording days.
    The model learns generalized features on Session 1 of Subject Group A, and evaluates verification on Subject Group B.

    Args:
        x_s1 (np.ndarray): Input ECG signals from Session 1.
        y_s1 (np.ndarray): Labels for Session 1.
        x_s2 (np.ndarray): Input ECG signals from Session 2.
        y_s2 (np.ndarray): Labels for Session 2.
        model_class (nn.Module): The PyTorch model architecture class to instantiate.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Number of samples per training batch.
        lr (float): Learning rate for the Adam optimizer.
        test_split (float): Fraction of unique SUBJECTS to isolate for the Group B tests.
        val_split (float): Fraction of Group A subjects to use for early stopping validation.
        num_pairs (int): Total number of Genuine and Impostor pairs to generate for evaluation.
        sampling_mode (str): Logic used to pair beats together.
            Options: ['all', 'balanced', 'random']
        seed (int): Random seed for reproducibility.
        device (str): Computation device ('cuda', 'cpu', or 'auto').
        visualize (bool): If True, generates t-SNE scatter plots of the unseen temporal embeddings.
        use_template (bool):
            - False: Evaluates raw temporal space (Group B's Session 2 paired vs Group B's Session 2).
            - True: Simulates Authentication (Group B's Session 2 probes matched vs Group B's Session 1 templates).
        template_fusion_method (str): Logic used to create Session 1 templates.
            Options: ['mean', 'median', 'trimmed_mean', 'representative',
            'soft_centrality', 'geometric_median', 'none']
        template_size (int, optional): Number of Session 1 beats used for enrollment. None uses all available.
        matching_method (str): Distance/Similarity metric used to score the pairs.
            Options: ['cosine', 'euclidean', 'manhattan', 'correlation']
        outlier_filtering_on_train (bool): Apply SQI filtering to Session 1 data.
        outlier_filtering_on_test (bool): Apply SQI filtering to Session 2 data.
        sqi_s1 (str or np.ndarray): SQI calculation method or pre-computed array for Session 1.
        sqi_s2 (str or np.ndarray): SQI calculation method or pre-computed array for Session 2.
        sqi_threshold (float): Absolute minimum SQI score required to keep a beat (0.0 to 1.0).
        sqi_keep_pct (float): Top percentage of beats to keep per subject after filtering.
        use_deployment_evaluation (bool): Uses unseen validation subjects from Group A to calculate a Global Threshold.
        save_results_and_settings (bool): If True, logs results and parameters to a text file.
        loader (object): Dataset loader instance (used for extracting metadata for logging).
        n_runs (int): Number of independent runs (with varying seeds) for statistical validation.
        _return_stats (bool): Internal flag used to pass data back during multi-seed recursion.

    Returns:
        tuple: (EER, AUC, d-prime, TAR @ 0.1% FAR)
               If n_runs > 1, returns tuples of (Mean, Std_Dev) for all four metrics.
    """

    _validate_deployment_evaluation(
        use_deployment_evaluation,
        val_split,
        "Subject-Disjoint Cross-Session Verification",
    )
    
    # ====================================================
    # 0. Capture Hyperparameters for Logger & MULTI-RUN AGGREGATOR
    # ====================================================
    data_stats = {}
    hyperparams = {
        'epochs': epochs, 'batch_size': batch_size, 'learning_rate': lr, 'test_split': test_split, 'val_split': val_split, 
        'num_pairs': num_pairs, 'use_template': use_template, 'template_fusion_method': template_fusion_method,
        'template_size': template_size, 'matching_method': matching_method, 'outlier_filter_train': outlier_filtering_on_train, 
        'outlier_filter_test': outlier_filtering_on_test
    }

    if n_runs > 1:
        call_args = _prepare_multi_run_arguments(locals())
        for k in ['data_stats', 'hyperparams', 'call_args', 'intelligent_weight_loading']: call_args.pop(k, None)
        call_args.update({'n_runs': 1, '_return_stats': True, 'save_results_and_settings': False})
        base_seed = call_args.get('seed', 42)
        results = []
        
        print(f"\n[INFO] Starting Multi-Seed Execution ({n_runs} runs)...")
        for i in range(n_runs):
            call_args['seed'] = base_seed + i
            call_args['visualize'] = False 
            print(f"\n{'='*40}\n RUN {i+1}/{n_runs} (Seed: {call_args['seed']})\n{'='*40}")
            res, d_stats, h_params = run_subject_disjoint_cross_session_verification(**call_args) 
            results.append(res); data_stats = d_stats; hyperparams = h_params
                
        metrics_t = list(zip(*results))
        means, stds = [np.mean(m) for m in metrics_t], [np.std(m) for m in metrics_t]
        
        if save_results_and_settings:
            hyperparams['n_runs'] = n_runs
            metrics_dict = {
                "EER": f"{means[0]:.4f} ± {stds[0]:.4f}", "AUC": f"{means[1]:.4f} ± {stds[1]:.4f}", 
                "d-prime": f"{means[2]:.4f} ± {stds[2]:.4f}", "TAR@0.1%FAR": f"{means[3]:.4f} ± {stds[3]:.4f}"
            }
            _log_experiment_results("Subject-Disjoint Cross-Session Verification", metrics_dict, data_stats, hyperparams, loader)
        return tuple(zip(means, stds))
    # ----------------------------

    _set_seed(seed); device = _get_device(device)
    task_title = "Subject-Disjoint Cross-Session Verification"
    mode_str = f"Template ({template_fusion_method}, S1 Enroll -> S2 Probe)" if use_template else "Cloud Pairs (S2 vs S2)"
    print(f"\n[TASK] {task_title} | Mode: {mode_str} | Match: {matching_method}")

    # ====================================================
    # 1. PREPARE & APPLY SQI FILTERS
    # ====================================================
    def _prepare_sqi(sqi_input, x_data, flag, name):
        if not flag: return None
        if sqi_input is None: return None
        if isinstance(sqi_input, str): return np.array(_compute_sqi(x_data, method=sqi_input))
        return np.array(sqi_input)

    sqi_s1 = _prepare_sqi(sqi_s1, x_s1, outlier_filtering_on_train, "Session 1")
    sqi_s2 = _prepare_sqi(sqi_s2, x_s2, outlier_filtering_on_test, "Session 2")

    if sqi_s1 is not None:
        print("\n[INFO] Filtering Session 1 (Representation & Enrollment)...")
        x_s1, y_s1 = _apply_outlier_filter(x_s1, y_s1, sqi_s1, sqi_threshold, sqi_keep_pct)

    if sqi_s2 is not None:
        print("\n[INFO] Filtering Session 2 (Probes)...")
        x_s2, y_s2 = _apply_outlier_filter(
            x_s2,
            y_s2,
            sqi_s2,
            absolute_threshold=sqi_threshold,
            keep_percentage=sqi_keep_pct,
            apply_subject_ranking=False,
        )

    # ====================================================
    # 2. INTERSECT AND SPLIT SUBJECTS
    # ====================================================
    common_subs = sorted(list(set(y_s1).intersection(set(y_s2))))
    
    if len(common_subs) < 2: 
        raise ValueError("[ERROR] Not enough common subjects across sessions after filtering.")
        
    train_subs_full, test_subs = train_test_split(common_subs, test_size=test_split, random_state=seed)
    
    if val_split > 0.0:
        train_subs, val_subs = train_test_split(train_subs_full, test_size=val_split, random_state=seed)
        print(f"Subject Split: Train={len(train_subs)}, Val={len(val_subs)}, Test={len(test_subs)}")
    else:
        train_subs = train_subs_full
        val_subs = []
        print(f"Subject Split: Train={len(train_subs)}, Val=0, Test={len(test_subs)}")

    X_train, Y_train = x_s1[np.isin(y_s1, train_subs)], y_s1[np.isin(y_s1, train_subs)]
    
    # STRICT TEMPORAL ISOLATION: Validation uses ONLY Session 1 data
    X_val_s1, Y_val_s1 = (x_s1[np.isin(y_s1, val_subs)], y_s1[np.isin(y_s1, val_subs)]) if len(val_subs) > 0 else (None, None)
    
    X_enroll, Y_enroll = x_s1[np.isin(y_s1, test_subs)], y_s1[np.isin(y_s1, test_subs)]
    X_probe, Y_probe = x_s2[np.isin(y_s2, test_subs)], y_s2[np.isin(y_s2, test_subs)]

    # ====================================================
    # 3. ENCODE LABELS (CRITICAL FIX FOR PYTORCH TENSORS)
    # ====================================================
    y_train_enc, train_classes = _encode_labels(Y_train)
    num_train_classes = len(train_classes)
    
    if len(val_subs) > 0:
        val_map = {sub: i for i, sub in enumerate(val_subs)}
        y_val_s1_enc = np.array([val_map[s] for s in Y_val_s1])
    else:
        y_val_s1_enc = None

    test_map = {sub: i for i, sub in enumerate(test_subs)}
    y_enroll_enc = np.array([test_map[s] for s in Y_enroll])
    y_probe_enc = np.array([test_map[s] for s in Y_probe])

    # ====================================================
    # 4. LOADERS & CUSTOM TRAINING LOOP
    # ====================================================
    # Create a Validation split from the SEEN Training subjects
    # This gives us the smooth Cross-Entropy loss anchor for the composite metric
    if val_split > 0.0:
        X_tr, X_val_seen, y_tr, y_val_seen = train_test_split(
            X_train, y_train_enc, test_size=val_split, stratify=y_train_enc, random_state=seed
        )
        val_loader_seen = _make_loader(X_val_seen, y_val_seen, batch_size, shuffle=False)
    else:
        X_tr, y_tr = X_train, y_train_enc
        val_loader_seen = None

    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader_s1 = _make_loader(X_val_s1, y_val_s1_enc, batch_size, shuffle=False) if X_val_s1 is not None else None
    
    # UNSEEN Validation now strictly passes the Session 1 loader only (Intra-Session check)
    val_loader_unseen = val_loader_s1
    
    model = model_class(in_channels=_detect_channels(x_s1), num_classes=num_train_classes, include_top=True).to(device)
    
    if intelligent_weight_loading:
        from utils import CacheManager
        cache = CacheManager()
        train_config = {
            "training_regime": "cross_session_subject_disjoint",
            "model": model_class.__name__, "epochs": epochs, "batch_size": batch_size, "lr": lr, 
            "val_split": val_split, "seed": seed, "outlier_train": outlier_filtering_on_train, 
            "sqi_thresh": sqi_threshold, "classes": num_train_classes, "data_shape": X_tr.shape,
            "matching_method": matching_method # Affects early stopping EER!
        }

        train_config = _build_weight_cache_config(
            loader,
            train_config,
        )

        cached_model, uid = cache.get_weight_cache(train_config, model, device)
        if cached_model:
            print(f"\n[INFO] Loaded pre-trained weights (Hash: {uid}). Skipping training!")
            model = cached_model
            model.actual_epochs = epochs
        else:
            print(f"\n[INFO] Training new Subject-Disjoint Cross-Session model (Hash: {uid})...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
            model = _run_train_loop_unseen_subjects(
                model=model, train_loader=train_loader, val_loader_seen=val_loader_seen, 
                val_loader_unseen=val_loader_unseen, optimizer=optimizer, criterion=criterion, 
                device=device, epochs=epochs, matching_method=matching_method, patience=40, lr_patience=15
            )
            cache.save_weight_cache(model, train_config, uid)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Single line execution using the Composite Metric loop!
        model = _run_train_loop_unseen_subjects(
            model=model, 
            train_loader=train_loader, 
            val_loader_seen=val_loader_seen, 
            val_loader_unseen=val_loader_unseen, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device, 
            epochs=epochs, 
            matching_method=matching_method, 
            patience=40,       # Max epochs to wait for composite score improvement
            lr_patience=15     # Epochs to wait before halving Learning Rate
        )

    # ====================================================
    # 5. MODEL CALIBRATION (Optional)
    # ====================================================
    model.include_top = False
    
    if use_deployment_evaluation:
        print("\n[INFO] --- DEPLOYMENT THRESHOLD CALIBRATION ---")
        calib_loader = val_loader_s1
        calib_name = "Unseen Validation (Session 1)"
            
        print(f"[INFO] Extracting features for Calibration ({calib_name})...")
        calib_emb_s1, calib_lab_s1 = _get_embeddings(model, calib_loader, device)
        
        # Calibration relies entirely on Session 1 features
        print(f"[INFO] Generating Calibration Pairs to find Global Threshold...")
        calib_scores, calib_pair_labels = _generate_pairs(
            embeddings1=calib_emb_s1, labels1=calib_lab_s1, 
            embeddings2=None, labels2=None,
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
        global_threshold = _find_optimal_threshold(calib_scores, calib_pair_labels)
        print(f"[INFO] Optimal Global Threshold Found: {global_threshold:.4f}")

    # ====================================================
    # 6. EVALUATION STRATEGY ON UNSEEN TEST SUBJECTS
    # ====================================================
    probe_loader = _make_loader(X_probe, y_probe_enc, batch_size, shuffle=False)
    emb_probe, lab_probe = _get_embeddings(model, probe_loader, device)

    if not use_template:
        print(f"[INFO] Bypassing Templates. Generating pairs entirely from Session 2 for Unseen Subjects...")
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, labels1=lab_probe, 
            embeddings2=None, labels2=None, 
            num_pairs=num_pairs, sampling_mode=sampling_mode, matching_method=matching_method
        )
    else:
        print(f"[INFO] Building Enrollment Templates for Unseen Subjects from Session 1...")
        enroll_loader = _make_loader(X_enroll, y_enroll_enc, batch_size, shuffle=False)
        emb_enroll, lab_enroll = _get_embeddings(model, enroll_loader, device)
        
        templates, temp_labels = _create_templates(
            emb_enroll, lab_enroll, method=template_fusion_method, max_beats=template_size
        )
        
        scores, labels_pair = _generate_pairs(
            embeddings1=emb_probe, # Session 2 Probes
            labels1=lab_probe, 
            embeddings2=templates, # Session 1 Templates
            labels2=temp_labels, 
            num_pairs=num_pairs, 
            sampling_mode=sampling_mode, 
            matching_method=matching_method
        )
        
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_probe, lab_probe, title="Disjoint Cross-Session Embeddings (T-SNE)")

    if use_deployment_evaluation:
        _evaluate_with_global_threshold(scores, labels_pair, global_threshold)

    eer, auc_val, dprime, tar = _compute_metrics_verification(scores, labels_pair)

    # Update hyperparams dictionary dynamically using the model's tracked epochs
    actual_ep = getattr(model, 'actual_epochs', epochs)
    hyperparams['epochs'] = f"{epochs} (stopped at {actual_ep})" if actual_ep < epochs else epochs

    data_stats = {
        "Train Subjects": len(train_subs),
        "Test Subjects": len(test_subs),
        "Train (S1) Samples": len(X_train),
        "Enrollment (S1) Samples": len(X_enroll),
        "Probe (S2) Samples": len(X_probe),
    }

    data_stats.update(
        _get_verification_pair_statistics(
            labels_pair,
            target_far=0.001,
        )
    )

    if _return_stats:
        return (
            eer,
            auc_val,
            dprime,
            tar,
        ), data_stats, hyperparams

    if save_results_and_settings:
        _log_experiment_results(
            task_title,
            {
                "EER": eer,
                "AUC": auc_val,
                "d-prime": dprime,
                "TAR@0.1%FAR": tar,
            },
            data_stats,
            hyperparams,
            loader,
        )

    return eer, auc_val, dprime, tar
# =============================================================================
