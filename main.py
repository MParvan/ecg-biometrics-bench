import argparse
import sys
import numpy as np
import yaml
from pathlib import Path
import run
import utils

# Import Dataset Loaders
from load_dataset import (
    load_ecgid_dataset, load_heartprint_dataset, load_ptb_dataset, 
    load_cybhi_dataset, load_mitbih_dataset, load_nsrdb_dataset, load_ptbxl_dataset
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
from models import DeepECG, ResNet1D, RNN_ECG, HybridCNNLSTM, ECGTransformer

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

CONFIG_KEY_ALIASES = {
    # Backward-compatible name used by experiment_settings.yaml
    "save_results_and_settings": "save_results",
}


def apply_yaml_config(args, parser):
    """
    Apply experiment settings from a YAML file.

    YAML values override argparse defaults and explicitly supplied CLI values.
    Unknown YAML keys raise an error instead of being silently ignored.
    """
    if not args.config:
        return args

    config_path = Path(args.config)

    if not config_path.exists():
        parser.error(f"Configuration file not found: {config_path}")

    print(f"[INFO] Loading experiment parameters from YAML: {config_path.name}")

    with open(config_path, "r", encoding="utf-8") as file:
        yaml_cfg = yaml.safe_load(file) or {}

    if not isinstance(yaml_cfg, dict):
        parser.error("The YAML configuration must contain a key-value mapping.")

    unknown_keys = []

    for yaml_key, value in yaml_cfg.items():
        argument_name = CONFIG_KEY_ALIASES.get(yaml_key, yaml_key)

        if not hasattr(args, argument_name):
            unknown_keys.append(yaml_key)
            continue

        setattr(args, argument_name, value)

    if unknown_keys:
        parser.error(
            "Unknown configuration key(s): "
            + ", ".join(sorted(unknown_keys))
        )

    return args

def get_parser():
    # Use RawTextHelpFormatter to preserve beautiful line breaks in the help menu
    parser = argparse.ArgumentParser(
        description=(
            "========================================================================\n"
            "🫀 DEEP LEARNING ECG BIOMETRICS FRAMEWORK\n"
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
            "📖 EXAMPLES OF USAGE:\n"
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
    core_group.add_argument('--dataset', type=str, required=True, 
                            choices=['ecgid', 'ptb', 'mitbih', 'nsrdb', 'ptbxl', 'heartprint', 'cybhi'],
                            help="Target database to load.")
    core_group.add_argument('--task', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8],
                            help="Biometric Evaluation Task Number (1 to 8).")
    core_group.add_argument('--model', type=str, default='deepecg', 
                            choices=['deepecg', 'resnet1d', 'rnn', 'hybrid', 'transformer'],
                            help="Neural Network Architecture (default: deepecg).")
    core_group.add_argument('--config', type=str, default=None,
                            help="Path to a YAML file to automatically override default parameters.")
    
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
    data_group.add_argument('--session_for_single_session_evaluation', type=str, nargs='+',
                            help="Target session if running intra-session tasks (1-4) on multi-session datasets.")
    data_group.add_argument('--num_beats_to_merge', type=int, default=1,
                            help="Consecutive beats to fuse natively in the loader (default: 1).")
    data_group.add_argument('--signal_type', type=str, default='raw', choices=['raw', 'filtered'],
                            help="For ECG-ID (raw vs filtered channel).")

    # ----------------------------------------------------
    # TRAINING HYPERPARAMETERS
    # ----------------------------------------------------
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--epochs', type=int, default=150, help="Max epochs (default: 150).")
    train_group.add_argument('--batch_size', type=int, default=256, help="Batch size (default: 256).")
    train_group.add_argument('--lr', type=float, default=1e-3, help="Learning rate (default: 0.001).")
    train_group.add_argument('--test_split', type=float, default=0.2, help="Percentage for Test set (default: 0.2).")
    train_group.add_argument('--val_split', type=float, default=0.1, help="Percentage for Validation set (default: 0.1).")
    train_group.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    train_group.add_argument('--n_runs', type=int, default=1, help="Number of independent runs using consecutive random seeds.")

    # ----------------------------------------------------
    # EVALUATION & TEMPLATE SETTINGS
    # ----------------------------------------------------
    eval_group = parser.add_argument_group('Evaluation & Biometric Settings')
    eval_group.add_argument('--use_template', action='store_true', 
                            help="Enable Template Matching (strips Softmax). Required for Tasks 3, 4, 7, 8.")
    eval_group.add_argument('--template_fusion_method', type=str, default='mean', 
                            choices=['mean', 'median', 'trimmed_mean', 'representative', 'none'],
                            help="How to aggregate multiple enrollment beats into one vector.")
    eval_group.add_argument('--template_size', type=int, default=None,
                            help="Max number of beats to use for enrollment (None = use all).")
    eval_group.add_argument('--matching_method', type=str, default='cosine',
                            choices=['cosine', 'euclidean', 'manhattan', 'correlation'],
                            help="Distance metric for verification/identification.")
    eval_group.add_argument('--num_pairs', type=int, default=10000,
                            help="Number of pairs to generate for Verification Tasks (default: 10000).")
    eval_group.add_argument(
        '--sampling_mode',
        type=str,
        default='all',
        choices=['all', 'balanced', 'random'],
        help="Verification pair generation strategy."
    )

    eval_group.add_argument(
        '--probe_fusion_size',
        type=int,
        default=3,
        help="Number of probe scores to fuse for identification tasks."
    )

    eval_group.add_argument(
        '--use_deployment_evaluation',
        action='store_true',
        help="Calibrate a verification threshold on validation data and apply it to test data."
    )

    # ----------------------------------------------------
    # SQI & FILTERING SETTINGS
    # ----------------------------------------------------
    sqi_group = parser.add_argument_group('Signal Quality & Filtering')
    sqi_group.add_argument('--outlier_filtering_on_train', action='store_true',
                           help="Apply SQI filter to Enrollment/Train data.")
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

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # ==========================================
    # 0. YAML CONFIGURATION
    # ==========================================
    args = apply_yaml_config(args, parser)

    # ==========================================
    # 1. MODEL SELECTION
    # ==========================================
    model_mapping = {
        'deepecg': DeepECG,
        'resnet1d': ResNet1D,
        'rnn': RNN_ECG,
        'hybrid': HybridCNNLSTM,
        'transformer': ECGTransformer
    }
    selected_model_class = model_mapping[args.model.lower()]

    # ==========================================
    # 2. DATASET INSTANTIATION
    # ==========================================
    # Build a kwargs dictionary dynamically, omitting None values
    loader_kwargs = {
        'data_split_mode': args.data_split_mode,
        'num_beats_to_merge': args.num_beats_to_merge
    }
    
    if args.train_sessions: loader_kwargs['train_sessions'] = args.train_sessions
    if args.enroll_sessions: loader_kwargs['enroll_sessions'] = args.enroll_sessions
    if args.probe_sessions: loader_kwargs['probe_sessions'] = args.probe_sessions
    if args.session_for_single_session_evaluation: 
        loader_kwargs['session_for_single_session_evaluation'] = args.session_for_single_session_evaluation
        
    if args.dataset == 'ecgid': loader_kwargs['signal_type'] = args.signal_type

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

    # ==========================================
    # 3. DATA EXTRACTION LOGIC
    # ==========================================
    if args.intelligent_data_loading:
        from utils import CacheManager
        cache = CacheManager()
        data_config = {
            "dataset": args.dataset, "split_mode": args.data_split_mode,
            "num_beats": args.num_beats_to_merge, "preprocessing": getattr(loader, 'prep_params', {}),
            "train_sessions": args.train_sessions, "test_sessions": args.probe_sessions
        }
        
        # Tasks 1 to 4: Intra-Session
        if args.task in [1, 2, 3, 4]:
            data_config["task_type"] = "intra_session"
            cached_data, uid = cache.get_data_cache(data_config)
            
            if cached_data:
                print(f"\n[INFO] Loaded precomputed data from cache (Hash: {uid})")
                x, y = cached_data['x'], cached_data['y']
            else:
                if args.dataset in ['cybhi', 'heartprint']:
                    x, y = loader.load_session("train")
                else:
                    if args.data_split_mode in ["all-available", "single-session"]:
                        x, y = loader.load_all_data()
                    else:
                        x, y = loader.load_session("train")
                cache.save_data_cache({'x': x, 'y': y}, data_config, uid)

            print(f"\n[INFO] Data Loaded: X={x.shape}, Y={y.shape}")
            if x.shape[0] == 0:
                print("[ERROR] No data returned from loader. Check your session configs.")
                sys.exit(1)

        # Tasks 5 to 8: Cross-Session
        elif args.task in [5, 6, 7, 8]:
            data_config["task_type"] = "cross_session"
            cached_data, uid = cache.get_data_cache(data_config)
            
            if cached_data:
                print(f"\n[INFO] Loaded precomputed cross-session data from cache (Hash: {uid})")
                x_s1, y_s1 = cached_data['x_s1'], cached_data['y_s1']
                x_s2, y_s2 = cached_data['x_s2'], cached_data['y_s2']
            else:
                x_s1, y_s1 = loader.load_session("train")
                x_s2, y_s2 = loader.load_session("test")
                cache.save_data_cache({'x_s1': x_s1, 'y_s1': y_s1, 'x_s2': x_s2, 'y_s2': y_s2}, data_config, uid)

            print(f"\n[INFO] Session 1 (Enroll) Loaded: X={x_s1.shape}, Y={y_s1.shape}")
            print(f"[INFO] Session 2 (Probe) Loaded:  X={x_s2.shape}, Y={y_s2.shape}")
            if x_s1.shape[0] == 0 or x_s2.shape[0] == 0:
                print("[ERROR] One or both cross-session arrays are empty. Check your parameters.")
                sys.exit(1)

    else:
        # Tasks 1 to 4: Intra-Session or Single Array operations
        if args.task in [1, 2, 3, 4]:
            if args.dataset in ['cybhi', 'heartprint']:
                x, y = loader.load_session("train")
            else:
                # EXPLICIT ROUTING: No more try/except guessing!
                if args.data_split_mode in ["all-available", "single-session"]:
                    x, y = loader.load_all_data()
                else:
                    x, y = loader.load_session("train")

            print(f"\n[INFO] Data Loaded: X={x.shape}, Y={y.shape}")
            if x.shape[0] == 0:
                print("[ERROR] No data returned from loader. Check your session configs.")
                sys.exit(1)
                
        # Tasks 5 to 8: Cross-Session (Requires S1 and S2 arrays)
        elif args.task in [5, 6, 7, 8]:
            x_s1, y_s1 = loader.load_session("train")
            x_s2, y_s2 = loader.load_session("test")
            
            print(f"\n[INFO] Session 1 (Enroll) Loaded: X={x_s1.shape}, Y={y_s1.shape}")
            print(f"[INFO] Session 2 (Probe) Loaded:  X={x_s2.shape}, Y={y_s2.shape}")
            if x_s1.shape[0] == 0 or x_s2.shape[0] == 0:
                print("[ERROR] One or both cross-session arrays are empty. Check your parameters.")
                sys.exit(1)


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
        'intelligent_weight_loading': args.intelligent_weight_loading
    }

    # ==========================================
    # 5. EXECUTE THE SELECTED TASK
    # ==========================================
    print("=" * 70)
    print(f"🚀 EXECUTING TASK {args.task} ON {args.dataset.upper()} USING {args.model.upper()}")
    print("=" * 70)

    try:
        # TASK 1: Closed-Set Identification
        if args.task == 1:
            run_closed_set_identification(
                x, y, 
                test_split=args.test_split, 
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                **common_args
            )

        # TASK 2: Verification
        elif args.task == 2:
            run_closed_set_verification(
                x, y, 
                test_split=args.test_split,
                num_pairs=args.num_pairs,
                sampling_mode=args.sampling_mode,
                use_deployment_evaluation=args.use_deployment_evaluation,
                sqi_scores=args.sqi_method,
                **common_args
            )

        # TASK 3: Subject-Disjoint Identification
        elif args.task == 3:
            run_subject_disjoint_identification(
                x, y, 
                test_split=args.test_split,
                probe_fusion_size=args.probe_fusion_size,
                sqi_scores=args.sqi_method,
                **common_args
            )

        # TASK 4: Subject-Disjoint Verification
        elif args.task == 4:
            run_subject_disjoint_verification(
                x, y, 
                test_split=args.test_split,
                num_pairs=args.num_pairs,
                sampling_mode=args.sampling_mode,
                use_deployment_evaluation=args.use_deployment_evaluation,
                sqi_scores=args.sqi_method,
                **common_args
            )

        # TASK 5: Cross-Session Identification
        elif args.task == 5:
            run_cross_session_identification(
                x_s1, y_s1, x_s2, y_s2, 
                probe_fusion_size=args.probe_fusion_size,
                sqi_train=args.sqi_method,
                sqi_test=args.sqi_method,
                **common_args
            )

        # TASK 6: Cross-Session Verification
        elif args.task == 6:
            run_cross_session_verification(
                x_s1, y_s1, x_s2, y_s2, 
                num_pairs=args.num_pairs,
                sampling_mode=args.sampling_mode,
                use_deployment_evaluation=args.use_deployment_evaluation,
                sqi_train=args.sqi_method,
                sqi_test=args.sqi_method,
                **common_args
            )

        # TASK 7: Subject-Disjoint Cross-Session ID
        elif args.task == 7:
            run_subject_disjoint_cross_session_identification(
                x_s1, y_s1, x_s2, y_s2, 
                test_split=args.test_split,
                probe_fusion_size=args.probe_fusion_size,
                sqi_s1=args.sqi_method,
                sqi_s2=args.sqi_method,
                **common_args
            )

        # TASK 8: Subject-Disjoint Cross-Session Verification
        elif args.task == 8:
            run_subject_disjoint_cross_session_verification(
                x_s1, y_s1, x_s2, y_s2, 
                test_split=args.test_split,
                num_pairs=args.num_pairs,
                sampling_mode=args.sampling_mode,
                use_deployment_evaluation=args.use_deployment_evaluation,
                sqi_s1=args.sqi_method,
                sqi_s2=args.sqi_method,
                **common_args
            )

        print("\n[SUCCESS] Pipeline execution complete.")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
