import os
import sys

# --- 1. PREVENT WINDOWS & FIX WHITE PLOTS ---
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# GLOBAL VARIABLE to store the filename for the next plot
CURRENT_PLOT_PATH = None

def custom_show(*args, **kwargs):
    """
    Monkey-patch function: Replaces plt.show().
    Instead of showing, it saves the current figure to CURRENT_PLOT_PATH.
    """
    global CURRENT_PLOT_PATH
    if CURRENT_PLOT_PATH:
        # Save the figure before it gets cleared
        plt.savefig(CURRENT_PLOT_PATH, bbox_inches='tight', dpi=300)
        print(f"    [Plot Saved]: {CURRENT_PLOT_PATH}")
        CURRENT_PLOT_PATH = None # Reset for next task
    plt.close() # Clear memory

# Overwrite plt.show with our custom saver
plt.show = custom_show
# --------------------------------------------

import torch
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import Project Modules
from load_dataset import load_cybhi_dataset
from run import *
from models import DeepECG, ResNet1D

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
CONFIG = {
    "DATASET": "CYBHi",
    "EPOCHS": 2,
    "BATCH_SIZE": 512,
    "NUM_BEATS": 3,
    "MODEL_CLASS": DeepECG,
    "LR": 0.001,
    "VISUALIZE": True,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "SEED": 42
}

# Output Directories
RESULTS_DIR = os.path.join(parent_dir, "results", "cybhi_final")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Log File
LOG_FILE = os.path.join(RESULTS_DIR, "cybhi_results_log.txt")

# Set Seeds
torch.manual_seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["SEED"])

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================
def log_header():
    with open(LOG_FILE, "w") as f:
        f.write("="*60 + "\n")
        f.write(f"EXPERIMENT LOG: {CONFIG['DATASET']}\n")
        f.write("="*60 + "\n")
        f.write("HYPERPARAMETERS:\n")
        for k, v in CONFIG.items():
            f.write(f"  {k}: {v}\n")
        f.write("="*60 + "\n\n")

def log_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n{title}\n{'='*60}\n")

def log_metric(task_name, result_str, shapes=None):
    entry = f"{task_name}\n"
    if shapes: entry += f"    Data Shapes: {shapes}\n"
    entry += f"    Result: {result_str}\n"
    entry += "-"*40 + "\n"

    print(f"  > {task_name}")
    if shapes: print(f"    Shapes: {shapes}")
    print(f"    Result: {result_str}")

    with open(LOG_FILE, "a") as f:
        f.write(entry)

def set_plot_filename(filename_suffix):
    """Sets the path where the NEXT plt.show() call will save the image."""
    global CURRENT_PLOT_PATH
    if CONFIG["VISUALIZE"]:
        CURRENT_PLOT_PATH = os.path.join(FIGURES_DIR, f"cybhi_{filename_suffix}.png")
    else:
        CURRENT_PLOT_PATH = None

# =============================================================================
# 4. PHASE 1: RANDOM SPLIT BASELINES
# =============================================================================
def run_random_split_phase():
    log_section("PHASE 1: RANDOM SPLIT BASELINES")

    # Load All Data
    # 'long_term' subset is used for random split experiments
    loader = load_cybhi_dataset(num_beats=CONFIG["NUM_BEATS"], subset='long-term', cleanup_zip=False)
    x_all, y_all = loader.load_all_sessions()
    shapes = {"All_Data": x_all.shape}

    # --- Task 1: Closed-Set Identification ---
    set_plot_filename("task1_random_id_cm")
    r1, r5 = run_closed_set_identification(
        x_all, y_all, CONFIG["MODEL_CLASS"], 
        epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
        device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    )
    log_metric("Task 1: Closed-Set ID", f"Rank-1: {r1:.4f}, Rank-5: {r5:.4f}", shapes)

    # --- Task 2: Verification (Closed-Set) ---
    set_plot_filename("task2_random_ver_roc")
    eer, auc, dprime, tar = run_verification(
        x_all, y_all, CONFIG["MODEL_CLASS"], 
        epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
        device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    )
    log_metric("Task 2: Verification", f"EER: {eer:.4f}, AUC: {auc:.4f}, d_prime': {dprime:.4f}, TAR@0.1%FAR: {tar:.4f}", shapes)

    # --- Task 3: Subject-Disjoint Verification ---
    set_plot_filename("task3_random_sd_ver_roc")
    eer_sd, auc_sd, dprime_sd, tar_sd = run_subject_disjoint_verification(
        x_all, y_all, CONFIG["MODEL_CLASS"], 
        epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
        device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    )
    log_metric("Task 3: Subject-Disjoint Verif", f"EER: {eer_sd:.4f}, AUC: {auc_sd:.4f}, d_prime': {dprime_sd:.4f}, TAR@0.1%FAR: {tar_sd:.4f}", shapes)

    # Task 4: Subject-Disjoint Identification (1-NN Template Matching) - Optional
    # set_plot("task4_random_sd_id")
    # r1_sd, r5_sd = run_subject_disjoint_identification(
    # x_all, y_all, CONFIG["MODEL_CLASS"],
    # epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
    # device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    # )
    # log_metric("Task 4: Subject-Disjoint ID", f"Rank-1: {r1_sd:.4f}, Rank-5: {r5_sd:.4f}", shapes)

# =============================================================================
# 5. PHASE 2: BIOMETRIC REGIMES
# =============================================================================
def run_regime_phase(regime_name, subset, file_suffix):
    log_section(f"PHASE 2: {regime_name}")

    # 1. Load Data
    loader = load_cybhi_dataset(num_beats=CONFIG["NUM_BEATS"], subset=subset, cleanup_zip=False)
    x_enr, y_enr = loader.load_session("Session_1")
    x_prb, y_prb = loader.load_session("Session_2")

    if len(x_enr) == 0 or len(x_prb) == 0:
        print(f"[WARN] Insufficient data for {regime_name}. Skipping.")
        return

    shapes = {"Enroll": x_enr.shape, "Probe": x_prb.shape}

    # 2. Cross-Session Identification
    set_plot_filename(f"{file_suffix}_id_cm")
    r1, r5 = run_cross_session_identification(
        x_enr, y_enr, x_prb, y_prb, CONFIG["MODEL_CLASS"],
        epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
        device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    )
    log_metric(f"{regime_name} - Identification", f"Rank-1: {r1:.4f}, Rank-5: {r5:.4f}", shapes)

    # 3. Cross-Session Verification
    set_plot_filename(f"{file_suffix}_ver_roc")
    eer, auc, dprime, tar = run_cross_session_verification(
        x_enr, y_enr, x_prb, y_prb, CONFIG["MODEL_CLASS"],
        epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"],
        device=CONFIG["DEVICE"], visualize=CONFIG["VISUALIZE"]
    )
    log_metric(f"{regime_name} - Verification", f"EER: {eer:.4f}, AUC: {auc:.4f}, d_prime': {dprime:.4f}, TAR@0.1%FAR: {tar:.4f}", shapes)

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    try:
        log_header()
        
        # --- Baseline ---
        run_random_split_phase()

        # --- Regime A: Short-Term ---
        run_regime_phase(
            regime_name="Regime A: Short-Term (State Robustness)",
            subset="short-term",
            file_suffix="regimeA_short"
        )

        # --- Regime B: Long-Term ---
        run_regime_phase(
            regime_name="Regime B: Long-Term (Time Stability)",
            subset="long-term",
            file_suffix="regimeB_long"
        )

        print("\n" + "="*60)
        print("[SUCCESS] All experiments completed.")
        print(f"Results Log: {LOG_FILE}")
        print(f"Figures:     {FIGURES_DIR}")
        print("="*60)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Script failed: {e}")
        import traceback
        traceback.print_exc()