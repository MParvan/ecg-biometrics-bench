# run.py
# -----------------------------------------------------------------------------
# UNIFIED TRAINING & EVALUATION UTILITY FOR ECG BIOMETRICS
# -----------------------------------------------------------------------------
# This module handles the core deep learning and biometric evaluation logic.
# It supports:
#   1. Closed-Set Identification (Who is this person?)
#   2. Subject-Disjoint Identification (Can we identify new people via templates?)
#   3. Verification (Is this person who they claim to be? - Random Split)
#   4. Subject-Disjoint Verification (Generalization to unseen subjects)
#   5. Cross-Session Identification (Temporal Robustness)
#   6. Cross-Session Verification (Temporal Robustness)
#
# METRICS:
#   - ID: Rank-1 Accuracy, Rank-5 Accuracy (CMC)
#   - Verif: EER, AUC, d-prime, TAR @ 0.1% FAR
# -----------------------------------------------------------------------------

import numpy as np
import random
import collections
from typing import Dict, Any, Optional, Tuple, List, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from visualizations import Visualizer

# =============================================================================
# 1. UTILITIES & SETUP
# =============================================================================
def _set_seed(seed: int = 42):
    """Ensures reproducibility across Numpy, Random, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _get_device(device: Optional[str] = None) -> str:
    """Auto-selects CUDA if available, unless specified otherwise."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def _encode_labels(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes string/arbitrary labels into 0..N-1 integers for CrossEntropyLoss.
    Returns: (encoded_labels, original_classes)
    """
    classes, y_enc = np.unique(y, return_inverse=True)
    return y_enc.astype(np.int64), classes

def _detect_channels(x: np.ndarray) -> int:
    """Auto-detects input channels (1 for univariate, 12 for standard ECG)."""
    if x.ndim == 2: return 1
    elif x.ndim == 3: return x.shape[1]
    else: raise ValueError(f"Unexpected input shape: {x.shape}")

def _make_loader(x, y, batch_size, shuffle=True, device='cpu'):
    """Creates a PyTorch DataLoader."""
    x_t = torch.from_numpy(x).float()
    if x_t.ndim == 2: x_t = x_t.unsqueeze(1) # Add channel dim if missing
    
    # If y is None (e.g., pure inference), create dummy labels
    if y is not None: 
        y_t = torch.from_numpy(y).long()
        ds = TensorDataset(x_t, y_t)
    else: 
        ds = TensorDataset(x_t, torch.zeros(len(x_t)))
        
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# =============================================================================
# 2. CORE TRAINING & INFERENCE LOOPS
# =============================================================================
def _train_epoch(model, loader, optimizer, criterion, device):
    """Standard PyTorch training loop for one epoch."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def _get_embeddings(model, loader, device):
    """
    Extracts deep features (embeddings) from the penultimate layer.
    Used for Verification tasks where we compare vector similarity.
    """
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            emb = model(xb) # Forward pass (model.include_top=False)
            embeddings.append(emb.cpu().numpy())
            labels.append(yb.numpy())
    return np.vstack(embeddings), np.concatenate(labels)

# =============================================================================
# 3. METRIC CALCULATION HELPERS
# =============================================================================
def _compute_metrics_verification(scores, labels_pair):
    """
    Calculates standard biometric verification metrics.
    
    Args:
        scores: Similarity scores (Cosine Similarity, -1 to 1).
        labels_pair: 1 for Genuine (Same ID), 0 for Imposter (Diff ID).
        
    Returns:
        tuple: (EER, AUC, d_prime, TAR@0.1%FAR)
    """
    fpr, tpr, thresholds = roc_curve(labels_pair, scores)
    roc_auc = auc(fpr, tpr)
    
    # 1. EER (Equal Error Rate): Where False Accept Rate = False Reject Rate
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except:
        eer = 1.0 # Fail safe

    # 2. d-prime (Decidability Index): Separation between Genuine/Imposter distributions
    gen_scores = [s for s, l in zip(scores, labels_pair) if l == 1]
    imp_scores = [s for s, l in zip(scores, labels_pair) if l == 0]
    
    if len(gen_scores) > 0 and len(imp_scores) > 0:
        mu_gen, sigma_gen = np.mean(gen_scores), np.std(gen_scores)
        mu_imp, sigma_imp = np.mean(imp_scores), np.std(imp_scores)
        # Formula: |mu1 - mu2| / sqrt((var1 + var2)/2)
        d_prime = abs(mu_gen - mu_imp) / np.sqrt(0.5 * (sigma_gen**2 + sigma_imp**2) + 1e-10)
    else:
        d_prime = 0.0

    # 3. TAR @ FAR (Security Metric): Accuracy when False Accepts are locked at 0.1%
    target_far = 0.001 # 0.1%
    try:
        tar_at_far = interp1d(fpr, tpr)(target_far)
    except:
        tar_at_far = 0.0

    print(f"[RESULT] EER: {eer:.4f} | AUC: {roc_auc:.4f} | d': {d_prime:.4f} | TAR@0.1%FAR: {tar_at_far:.4f}")
    return eer, roc_auc, d_prime, tar_at_far

def _compute_metrics_identification(preds_probs, true_labels):
    """
    Calculates Rank-N Identification accuracy.
    
    Args:
        preds_probs: Softmax probabilities (N_samples, N_classes).
        true_labels: Ground truth integers (N_samples).
        
    Returns:
        tuple: (Rank-1 Accuracy, Rank-5 Accuracy)
    """
    # Sort predictions by probability (descending)
    # argsort sorts ascending, so we take reverse slices
    top_k_preds = np.argsort(preds_probs, axis=1)[:, ::-1] 
    
    # Rank 1: Is the top prediction correct?
    rank1 = np.mean(top_k_preds[:, 0] == true_labels)
    
    # Rank 5: Is the correct label in the top 5 predictions?
    # Handle case where N_classes < 5
    k = min(5, preds_probs.shape[1])
    hits_rank5 = [1 if true_labels[i] in top_k_preds[i, :k] else 0 for i in range(len(true_labels))]
    rank5 = np.mean(hits_rank5)
    
    print(f"[RESULT] Rank-1 Acc: {rank1:.4f} | Rank-5 Acc: {rank5:.4f}")
    return rank1, rank5

# =============================================================================
# HELPER: PAIR GENERATION (Supports ALL, BALANCED, RANDOM)
# =============================================================================
def _generate_pairs(embeddings1, labels1, embeddings2=None, labels2=None, num_pairs=10000, sampling_mode="balanced"):
    """
    Generates similarity scores and ground truth labels for verification.
    
    MODES:
      - 'balanced': Creates 50% Genuine and 50% Imposter pairs (Prevents bias).
      - 'random': Picks pairs completely at random (May be heavily unbalanced).
      - 'all': Computes the full Similarity Matrix (Every possible pair).
      
    Args:
        embeddings1: Query set (Probe).
        labels1: Query labels.
        embeddings2: (Optional) Template set (Enrollment). If None, does Intra-session (emb1 vs emb1).
    """
    scores = []
    labels_pair = []
    
    is_cross_session = (embeddings2 is not None)
    
    if not is_cross_session:
        embeddings2 = embeddings1
        labels2 = labels1

    # --------------------------------------------------------
    # MODE A: ALL PAIRS (Full Matrix)
    # --------------------------------------------------------
    if sampling_mode == "all":
        print(f"[INFO] generating ALL pairs (Full Matrix evaluation)...")
        # Compute Sim Matrix (N x M)
        sim_matrix = np.dot(embeddings1, embeddings2.T)
        
        # Ground Truth Matrix (1 if same label, 0 if diff)
        truth_matrix = (labels1[:, None] == labels2[None, :]).astype(int)
        
        if is_cross_session:
            # Flatten everything (N * M pairs)
            scores = sim_matrix.flatten()
            labels_pair = truth_matrix.flatten()
        else:
            # Intra-session: We must remove self-comparisons (diagonal) and duplicates (lower triangle)
            # Use upper triangle indices (k=1 excludes diagonal)
            upper_tri = np.triu_indices(len(labels1), k=1)
            scores = sim_matrix[upper_tri]
            labels_pair = truth_matrix[upper_tri]
            
    # --------------------------------------------------------
    # MODE B: BALANCED (Standard)
    # --------------------------------------------------------
    elif sampling_mode == "balanced":
        # Group indices by subject
        s1_idx = collections.defaultdict(list)
        s2_idx = collections.defaultdict(list)
        for i, l in enumerate(labels1): s1_idx[l].append(i)
        for i, l in enumerate(labels2): s2_idx[l].append(i)
        
        # Intersection of subjects (needed for Genuine pairs)
        common_subs = list(set(s1_idx.keys()) & set(s2_idx.keys()))
        if len(common_subs) < 2: return [], []

        # Genuine Pairs
        for _ in range(num_pairs // 2):
            subj = np.random.choice(common_subs)
            if is_cross_session:
                idx1 = np.random.choice(s1_idx[subj])
                idx2 = np.random.choice(s2_idx[subj])
            else:
                # Intra: Pick 2 different samples
                if len(s1_idx[subj]) < 2: continue
                idx1, idx2 = np.random.choice(s1_idx[subj], 2, replace=False)
                
            scores.append(np.dot(embeddings1[idx1], embeddings2[idx2]))
            labels_pair.append(1)

        # Imposter Pairs
        all_s1 = list(s1_idx.keys())
        all_s2 = list(s2_idx.keys())
        for _ in range(num_pairs // 2):
            s_a = np.random.choice(all_s1)
            # Pick s_b such that it is NOT s_a
            possible_b = [s for s in all_s2 if s != s_a]
            if not possible_b: continue
            s_b = np.random.choice(possible_b)
            
            idx1 = np.random.choice(s1_idx[s_a])
            idx2 = np.random.choice(s2_idx[s_b])
            scores.append(np.dot(embeddings1[idx1], embeddings2[idx2]))
            labels_pair.append(0)

    # --------------------------------------------------------
    # MODE C: RANDOM
    # --------------------------------------------------------
    elif sampling_mode == "random":
        indices1 = np.arange(len(labels1))
        indices2 = np.arange(len(labels2))
        for _ in range(num_pairs):
            i1 = np.random.choice(indices1)
            i2 = np.random.choice(indices2)
            
            # If intra-session, ensure we don't compare self to self
            if not is_cross_session and i1 == i2: continue
            
            scores.append(np.dot(embeddings1[i1], embeddings2[i2]))
            labels_pair.append(1 if labels1[i1] == labels2[i2] else 0)
            
    return np.array(scores), np.array(labels_pair)


# =============================================================================
# TASK 1: CLOSED-SET IDENTIFICATION
# =============================================================================
def run_closed_set_identification(x, y, model_class, epochs=30, batch_size=64, lr=1e-3, val_split=0.2, seed=42, device=None, visualize=False):
    """
    Standard Classification Task.
    Train on N subjects -> Test on same N subjects (different samples).
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Closed-Set Identification on {device}")
    
    y_enc, classes = _encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=val_split, stratify=y_enc, random_state=seed)
    
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    
    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    # Train
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")
        
    # Test
    model.eval()
    all_probs, all_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1) # Get probabilities
            all_probs.append(probs.cpu().numpy())
            all_trues.append(yb.cpu().numpy())
            
    all_probs = np.vstack(all_probs)
    all_trues = np.concatenate(all_trues)
    
    if visualize:
        viz = Visualizer()
        preds = np.argmax(all_probs, axis=1)
        viz.plot_confusion_matrix(all_trues, preds, normalize=True)

    return _compute_metrics_identification(all_probs, all_trues)

# =============================================================================
# TASK 2: SUBJECT-DISJOINT IDENTIFICATION (OPEN SET / TEMPLATE MATCHING)
# =============================================================================
def run_subject_disjoint_identification(x, y, model_class, train_subject_ratio=0.7, enrollment_beats=5, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
    """
    Identification on Unseen Subjects.
    Since Softmax can't handle new classes, we use Template Matching (1-NN).
    1. Train Feature Extractor on 'Train Subjects'.
    2. Enroll 'Test Subjects' using 'enrollment_beats'.
    3. Test remaining beats against Enrolled Templates.
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Subject-Disjoint ID (Template Matching) on {device}")
    
    subjects = np.unique(y)
    train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
    
    # 1. Train Feature Extractor
    mask_train = np.isin(y, train_subs)
    X_train, y_train = x[mask_train], y[mask_train]
    y_train_enc, _ = _encode_labels(y_train)
    
    train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
    model = model_class(in_channels=_detect_channels(x), num_classes=len(train_subs), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")
    
    # 2. Template Matching
    model.include_top = False # Embedding mode
    mask_test = np.isin(y, test_subs)
    X_test_all, y_test_all = x[mask_test], y[mask_test]
    X_enroll, y_enroll, X_query, y_query = [], [], [], []
    
    for sub in test_subs:
        sub_beats = X_test_all[y_test_all == sub]
        if len(sub_beats) <= enrollment_beats: continue
        X_enroll.append(sub_beats[:enrollment_beats]); y_enroll.extend([sub]*enrollment_beats)
        X_query.append(sub_beats[enrollment_beats:]); y_query.extend([sub]*(len(sub_beats)-enrollment_beats))
        
    enroll_loader = _make_loader(np.vstack(X_enroll), None, batch_size, shuffle=False)
    query_loader = _make_loader(np.vstack(X_query), None, batch_size, shuffle=False)
    emb_enroll, _ = _get_embeddings(model, enroll_loader, device)
    emb_query, _ = _get_embeddings(model, query_loader, device)
    
    # Create Prototypes (Mean embedding of enrollment shots)
    templates = []
    template_ids = np.unique(y_enroll)
    for sub in template_ids:
        idxs = np.where(np.array(y_enroll) == sub)[0]
        templates.append(np.mean(emb_enroll[idxs], axis=0))
    templates = np.array(templates)
    
    # Cosine Similarity (1-NN)
    # Sim Matrix: (N_query, N_templates)
    sim_matrix = np.dot(normalize(emb_query), normalize(templates).T)
    
    # Rank-1 Preds
    pred_indices = np.argmax(sim_matrix, axis=1)
    preds = template_ids[pred_indices]
    
    acc = np.mean(preds == np.array(y_query))
    print(f"[RESULT] Subject-Disjoint Rank-1 Acc: {acc:.4f}")
    
    if visualize:
        viz = Visualizer()
        # Visualize only a subset if confusion matrix is huge
        if len(template_ids) < 50:
            viz.plot_confusion_matrix(y_query, preds, normalize=True)
            
    return acc

# =============================================================================
# TASK 3: VERIFICATION (RANDOM SPLIT)
# =============================================================================
def run_verification(x, y, model_class, train_split=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
    """
    Standard Verification.
    Train and Test sets contain the SAME subjects (Closed-Set), but different beats.
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Verification (Closed-Set) on {device}")
    
    y_enc, classes = _encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=(1-train_split), stratify=y_enc, random_state=seed)
    
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    model = model_class(in_channels=_detect_channels(x), num_classes=len(classes), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")
    
    model.include_top = False
    test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
    emb_test, labels_test = _get_embeddings(model, test_loader, device)
    
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_test, labels_test, title="Verification Embeddings (T-SNE)")

    # Intra-session pair generation
    scores, labels_pair = _generate_pairs(normalize(emb_test), labels_test, None, None, num_pairs, sampling_mode)
    if len(scores) == 0: return 1.0, 0.5, 0.0, 0.0

    return _compute_metrics_verification(scores, labels_pair)

# =============================================================================
# TASK 4: SUBJECT-DISJOINT VERIFICATION
# =============================================================================
def run_subject_disjoint_verification(x, y, model_class, train_subject_ratio=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
    """
    Verification on Unseen Subjects.
    Train on Set A, Verify on Set B. This is the hardest verification task.
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Subject-Disjoint Verification on {device}")
    
    # Split by Subject ID
    subjects = np.unique(y)
    train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
    
    mask_train = np.isin(y, train_subs)
    X_train, y_train_raw = x[mask_train], y[mask_train]
    y_train_enc, _ = _encode_labels(y_train_raw)
    
    train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
    model = model_class(in_channels=_detect_channels(x), num_classes=len(train_subs), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")
    
    model.include_top = False
    mask_test = np.isin(y, test_subs)
    X_test, y_test_raw = x[mask_test], y[mask_test]
    test_loader = _make_loader(X_test, None, batch_size, shuffle=False)
    emb_test, _ = _get_embeddings(model, test_loader, device)

    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb_test, y_test_raw, title="Verification Embeddings (T-SNE)")
    
    scores, labels_pair = _generate_pairs(normalize(emb_test), y_test_raw, None, None, num_pairs, sampling_mode)
    if len(scores) == 0: return 1.0, 0.5, 0.0, 0.0
    
    return _compute_metrics_verification(scores, labels_pair)

# =============================================================================
# TASK 5: CROSS-SESSION IDENTIFICATION
# =============================================================================
def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
    """
    Train on Session 1 (Enrollment) -> Identify in Session 2 (Probe).
    Only evaluates on subjects present in BOTH sessions.
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Cross-Session Identification (Rank-1/Rank-5) on {device}")
    
    train_subs, test_subs = np.unique(y_train), np.unique(y_test)
    common_subs = np.intersect1d(train_subs, test_subs)
    if len(common_subs) == 0: raise ValueError("No overlapping subjects!")
    common_subs.sort()
    
    # Remap labels to 0..N for the common subset
    cls_map = {s: i for i, s in enumerate(common_subs)}
    
    mask_train = np.isin(y_train, common_subs)
    X_train_filt, y_train_filt = x_train[mask_train], y_train[mask_train]
    y_train_enc = np.array([cls_map[s] for s in y_train_filt])
    
    mask_test = np.isin(y_test, common_subs)
    X_test_filt, y_test_filt = x_test[mask_test], y_test[mask_test]
    y_test_enc = np.array([cls_map[s] for s in y_test_filt])
    
    train_loader = _make_loader(X_train_filt, y_train_enc, batch_size, shuffle=True)
    test_loader = _make_loader(X_test_filt, y_test_enc, batch_size, shuffle=False)
    
    model = model_class(in_channels=_detect_channels(x_train), num_classes=len(common_subs), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")
    
    model.eval()
    all_probs, all_trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.softmax(model(xb), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_trues.append(yb.numpy())
            
    all_probs = np.vstack(all_probs)
    all_trues = np.concatenate(all_trues)
    
    if visualize:
        viz = Visualizer()
        preds = np.argmax(all_probs, axis=1)
        viz.plot_confusion_matrix(all_trues, preds, normalize=True)
        
    return _compute_metrics_identification(all_probs, all_trues)

# =============================================================================
# TASK 6: CROSS-SESSION VERIFICATION
# =============================================================================
def run_cross_session_verification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
    """
    Train on S1 -> Generate Embeddings for S1 and S2 -> Pair S1 vs S2.
    """
    _set_seed(seed); device = _get_device(device)
    print(f"\n[TASK] Cross-Session Verification on {device}")
    
    # Train on Combined Data (or just S1) to learn identity features
    all_labels = np.concatenate([y_train, y_test])
    unique_labels = np.unique(all_labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    y_train_int = np.array([label_to_int[l] for l in y_train])
    
    train_loader = _make_loader(x_train, y_train_int, batch_size, shuffle=True)
    model = model_class(in_channels=_detect_channels(x_train), num_classes=len(label_to_int), include_top=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
    for ep in range(epochs): 
        loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {ep+1}/{epochs} | Loss: {loss:.4f}")

    model.include_top = False
    
    # Get Embeddings for S1 (Enrollment) and S2 (Probe)
    loader1 = _make_loader(x_train, None, batch_size, shuffle=False)
    emb1, _ = _get_embeddings(model, loader1, device)
    lab1 = y_train  # Use the original string labels
    
    loader2 = _make_loader(x_test, None, batch_size, shuffle=False)
    emb2, _ = _get_embeddings(model, loader2, device)
    lab2 = y_test   # Use the original string labels        
    model.include_top = False

    # Cross-session pair generation (Pass S1 and S2 separately)
    scores, labels_pair = _generate_pairs(normalize(emb1), lab1, normalize(emb2), lab2, num_pairs, sampling_mode)
    
    if len(scores) == 0: return 1.0, 0.5, 0.0, 0.0
    
    if visualize:
        viz = Visualizer()
        viz.plot_embeddings(emb2, lab2, title="Verification Embeddings (T-SNE)")

    return _compute_metrics_verification(scores, labels_pair)




# # run.py
# # ---------------------------------------------------
# # Unified Training & Evaluation Utility for ECG Biometrics
# # Supports:
# #   1. Closed-Set Identification (Classification)
# #   2. Subject-Disjoint Identification (Open-Set / Enrollment)
# #   3. Verification (Authentication / EER) - Random Split
# #   4. Subject-Disjoint Verification (EER) - Open Set
# #   5. Cross-Session Identification (Temporal Robustness)
# #   6. Cross-Session Verification (Temporal Robustness)
# # ---------------------------------------------------

# import numpy as np
# import random
# from typing import Dict, Any, Union, Callable, Optional, List, Tuple

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, confusion_matrix
# from sklearn.preprocessing import normalize
# from scipy.optimize import brentq
# from scipy.interpolate import interp1d

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# from visualizations import Visualizer

# def _set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# def _get_device(device: Optional[str] = None) -> str:
#     if device is None or device == "auto":
#         return "cuda" if torch.cuda.is_available() else "cpu"
#     return device

# def _encode_labels(y: np.ndarray):
#     classes, y_enc = np.unique(y, return_inverse=True)
#     return y_enc.astype(np.int64), classes

# def _detect_channels(x: np.ndarray) -> int:
#     if x.ndim == 2: return 1
#     elif x.ndim == 3: return x.shape[1]
#     else: raise ValueError(f"Unexpected input shape: {x.shape}")

# def _make_loader(x, y, batch_size, shuffle=True, device='cpu'):
#     x_t = torch.from_numpy(x).float()
#     if x_t.ndim == 2: x_t = x_t.unsqueeze(1)
#     if y is not None: y_t = torch.from_numpy(y).long(); ds = TensorDataset(x_t, y_t)
#     else: ds = TensorDataset(x_t, torch.zeros(len(x_t)))
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# def _train_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for xb, yb in loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * xb.size(0)
#     return total_loss / len(loader.dataset)

# def _get_embeddings(model, loader, device):
#     model.eval()
#     embeddings, labels = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb = xb.to(device)
#             emb = model(xb)
#             embeddings.append(emb.cpu().numpy())
#             labels.append(yb.numpy())
#     return np.vstack(embeddings), np.concatenate(labels)

# def _compute_eer(embeddings, labels, num_pairs=10000, sampling_mode="balanced"):
#     """
#     Computes EER and AUC using different pair sampling strategies.
    
#     Args:
#         embeddings: (N, D) array of feature vectors.
#         labels: (N,) array of subject IDs.
#         num_pairs: Number of pairs to generate (ignored if mode='all').
#         sampling_mode: 'balanced' (recommended), 'all', or 'random'.
#     """
#     import collections
#     from scipy.optimize import brentq
#     from scipy.interpolate import interp1d
#     from sklearn.metrics import roc_curve, auc

#     scores = []
#     labels_pair = []
    
#     # -------------------------------------------------------------------------
#     # MODE 1: ALL PAIRS (Similarity Matrix)
#     # -------------------------------------------------------------------------
#     if sampling_mode == "all":
#         print(f"[INFO] Computing ALL pairs (Full Similarity Matrix)...")
#         # Compute full NxN similarity matrix
#         sim_matrix = np.dot(embeddings, embeddings.T)
        
#         # Create ground truth matrix (1 if same label, 0 if diff)
#         # Broadcasting logic: labels[:, None] == labels[None, :]
#         truth_matrix = (labels[:, None] == labels[None, :]).astype(int)
        
#         # We only want the upper triangle (excluding diagonal) to avoid duplicates
#         # and self-comparisons (i.e., comparing A to A)
#         upper_tri_indices = np.triu_indices(len(labels), k=1)
        
#         scores = sim_matrix[upper_tri_indices]
#         labels_pair = truth_matrix[upper_tri_indices]
        
#         print(f"[INFO] Evaluated {len(scores)} unique pairs.")

#     # -------------------------------------------------------------------------
#     # MODE 2: BALANCED (50% Genuine / 50% Imposter) - STANDARD
#     # -------------------------------------------------------------------------
#     elif sampling_mode == "balanced":
#         print(f"[INFO] Generating {num_pairs} BALANCED pairs (50/50 split)...")
        
#         # Map labels to indices for fast lookup
#         class_indices = collections.defaultdict(list)
#         for idx, label in enumerate(labels):
#             class_indices[label].append(idx)
        
#         # A. Genuine Pairs (Positive)
#         # We need classes that have at least 2 samples to form a genuine pair
#         valid_genuine_classes = [c for c, idxs in class_indices.items() if len(idxs) >= 2]
        
#         if len(valid_genuine_classes) < 2:
#             print("[WARN] Not enough classes with multiple samples for balanced EER.")
#             return {"eer": 1.0, "auc": 0.5}

#         for _ in range(num_pairs // 2):
#             c = np.random.choice(valid_genuine_classes)
#             # Pick 2 different samples from the same class
#             idx1, idx2 = np.random.choice(class_indices[c], 2, replace=False)
#             scores.append(np.dot(embeddings[idx1], embeddings[idx2]))
#             labels_pair.append(1) # Genuine

#         # B. Imposter Pairs (Negative)
#         all_classes = list(class_indices.keys())
#         for _ in range(num_pairs // 2):
#             # Pick 2 different classes
#             c1, c2 = np.random.choice(all_classes, 2, replace=False)
#             idx1 = np.random.choice(class_indices[c1])
#             idx2 = np.random.choice(class_indices[c2])
#             scores.append(np.dot(embeddings[idx1], embeddings[idx2]))
#             labels_pair.append(0) # Imposter

#     # -------------------------------------------------------------------------
#     # MODE 3: RANDOM (Unbalanced)
#     # -------------------------------------------------------------------------
#     elif sampling_mode == "random":
#         print(f"[INFO] Generating {num_pairs} RANDOM pairs (likely imbalanced)...")
#         indices = np.arange(len(labels))
#         for _ in range(num_pairs):
#             i1, i2 = np.random.choice(indices, 2, replace=False)
#             scores.append(np.dot(embeddings[i1], embeddings[i2]))
#             labels_pair.append(1 if labels[i1] == labels[i2] else 0)

#     else:
#         raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

#     # -------------------------------------------------------------------------
#     # COMPUTE METRICS
#     # -------------------------------------------------------------------------
#     fpr, tpr, thresholds = roc_curve(labels_pair, scores)
#     roc_auc = auc(fpr, tpr)
    
#     try:
#         eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#     except Exception as e:
#         print(f"[WARN] EER calculation failed (likely perfect separation): {e}")
#         eer = 0.0

#     print(f"[RESULT] Mode: {sampling_mode.upper()} | EER: {eer:.4f} | AUC: {roc_auc:.4f}")
#     return {"eer": eer, "auc": roc_auc}

# # --- TASK 1: CLOSED-SET IDENTIFICATION ---
# def run_closed_set_identification(x, y, model_class, epochs=30, batch_size=64, lr=1e-3, val_split=0.2, seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Closed-Set Identification on {device}")
#     y_enc, classes = _encode_labels(y)
#     X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=val_split, stratify=y_enc, random_state=seed)
#     train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
#     test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(classes), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs):
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
        
#     model.eval()
#     all_preds, all_trues = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds = torch.argmax(model(xb), dim=1)
#             all_preds.append(preds.cpu().numpy()); all_trues.append(yb.cpu().numpy())
#     all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
#     acc = np.mean(all_preds == all_trues)
#     print(f"[RESULT] Closed-Set Accuracy: {acc:.4f}")

#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(all_trues, all_preds, normalize=True)

#     return acc

# # --- TASK 2: SUBJECT-DISJOINT IDENTIFICATION ---
# def run_subject_disjoint_identification(x, y, model_class, train_subject_ratio=0.7, enrollment_beats=5, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Subject-Disjoint Identification on {device}")
#     subjects = np.unique(y)
#     train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
    
#     # Train
#     mask_train = np.isin(y, train_subs)
#     X_train, y_train = x[mask_train], y[mask_train]
#     y_train_enc, _ = _encode_labels(y_train)
#     train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(train_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training feature extractor...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     # Test
#     print("[INFO] Phase 2: Enrollment & Matching...")
#     model.include_top = False
#     mask_test = np.isin(y, test_subs)
#     X_test_all, y_test_all = x[mask_test], y[mask_test]
#     X_enroll, y_enroll, X_query, y_query = [], [], [], []
    
#     for sub in test_subs:
#         sub_beats = X_test_all[y_test_all == sub]
#         if len(sub_beats) <= enrollment_beats: continue
#         X_enroll.append(sub_beats[:enrollment_beats]); y_enroll.extend([sub]*enrollment_beats)
#         X_query.append(sub_beats[enrollment_beats:]); y_query.extend([sub]*(len(sub_beats)-enrollment_beats))
        
#     enroll_loader = _make_loader(np.vstack(X_enroll), None, batch_size, shuffle=False)
#     query_loader = _make_loader(np.vstack(X_query), None, batch_size, shuffle=False)
#     emb_enroll, _ = _get_embeddings(model, enroll_loader, device)
#     emb_query, _ = _get_embeddings(model, query_loader, device)
    
#     templates, template_ids = [], []
#     for sub in np.unique(y_enroll):
#         idxs = np.where(np.array(y_enroll) == sub)[0]
#         templates.append(np.mean(emb_enroll[idxs], axis=0))
#         template_ids.append(sub)
    
#     sim_matrix = np.dot(normalize(emb_query), normalize(np.array(templates)).T)
#     preds = np.array(template_ids)[np.argmax(sim_matrix, axis=1)]
#     acc = np.mean(preds == np.array(y_query))
#     print(f"[RESULT] Subject-Disjoint Accuracy: {acc:.4f}")
#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(y_query, preds, normalize=True)
#     return acc

# # =============================================================================
# # TASK 3: VERIFICATION (RANDOM SPLIT / CLOSED-SET)
# # =============================================================================
# def run_verification(x, y, model_class, train_split=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     """
#     Performs 'Closed-Set' Verification (Seen Subjects).
    
#     SCENARIO:
#     The model is trained on a set of subjects (e.g., Alice, Bob) and tested on 
#     different recordings of the SAME subjects (Alice, Bob).
    
#     USE CASE:
#     - Unlocking a personal phone (the phone knows you, it just needs to verify it's you now).
#     - Biometric attendance systems for employees.
    
#     Args:
#         x: Input signals.
#         y: Labels.
#         sampling_mode: 'balanced' (50/50 +/-, recommended), 'all' (N*N matrix), or 'random'.
#     """
#     _set_seed(seed)
#     device = _get_device(device)
#     print(f"\n[TASK] Verification (Random Split / Closed-Set) on {device}")
    
#     # 1. Encode Labels (0, 1, 2...)
#     y_enc, classes = _encode_labels(y)
    
#     # 2. Random Split (Stratified)
#     # WARNING: This splits by SAMPLE. If a subject has 10 beats, 7 go to train, 3 to test.
#     # The model "sees" the subject during training.
#     X_train, X_test, y_train, y_test = train_test_split(
#         x, y_enc, test_size=(1-train_split), stratify=y_enc, random_state=seed
#     )
    
#     train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
#     in_ch = _detect_channels(x)
    
#     # 3. Model Setup (Includes Classification Head for Training)
#     model = model_class(in_channels=in_ch, num_classes=len(classes), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training feature extractor (Learning Identity)...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     # 4. Evaluation (Remove Classification Head)
#     print(f"[INFO] Phase 2: Computing EER using '{sampling_mode}' sampling...")
#     model.include_top = False # Switch to embedding mode
    
#     test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
#     emb_test, labels_test = _get_embeddings(model, test_loader, device)

#     if visualize:
#         viz = Visualizer()
#         # Plot T-SNE of the test embeddings
#         viz.plot_embeddings(emb_test, labels_test, title="Verification Embeddings (T-SNE)")
    
#     # Compute EER
#     # Note: We normalize embeddings to unit length for Cosine Similarity
#     return _compute_eer(normalize(emb_test), labels_test, num_pairs, sampling_mode=sampling_mode)


# # =============================================================================
# # TASK 4: SUBJECT-DISJOINT VERIFICATION (OPEN-SET)
# # =============================================================================
# def run_subject_disjoint_verification(x, y, model_class, train_subject_ratio=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     """
#     Performs 'Open-Set' Verification (Unseen Subjects).
    
#     SCENARIO:
#     The model is trained on a set of subjects (e.g., Alice, Bob) but tested on 
#     completely NEW subjects (e.g., Charlie, Dave) it has never seen before.
    
#     USE CASE:
#     - Universal Feature Extractors.
#     - Testing if the model learned "What makes an ECG unique?" rather than "What does Alice look like?"
#     - Large-scale surveillance or border control systems.
    
#     Args:
#         train_subject_ratio: % of subjects to use for training (e.g., 0.7 means 70% of people are in train).
#         sampling_mode: 'balanced' (recommended), 'all', or 'random'.
#     """
#     _set_seed(seed)
#     device = _get_device(device)
#     print(f"\n[TASK] Subject-Disjoint Verification (Open-Set) on {device}")
    
#     # 1. Split by SUBJECT ID (Not by sample)
#     subjects = np.unique(y)
#     train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
#     print(f"[INFO] Splitting: {len(train_subs)} Training Subjects vs {len(test_subs)} Test Subjects")
    
#     # 2. Create Train Set (Only Train Subjects)
#     mask_train = np.isin(y, train_subs)
#     X_train, y_train_raw = x[mask_train], y[mask_train]
#     y_train_enc, _ = _encode_labels(y_train_raw) # Encode 0..N for Softmax loss
    
#     train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
    
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(train_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training on Known Subjects...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     # 3. Create Test Set (Only New/Unseen Subjects)
#     print(f"[INFO] Phase 2: Computing EER on {len(test_subs)} Unseen Subjects...")
#     model.include_top = False # Remove classifier, we only want embeddings
    
#     mask_test = np.isin(y, test_subs)
#     X_test, y_test_raw = x[mask_test], y[mask_test]
    
#     # Note: We don't need encoded labels for testing, just the raw IDs to check equality
#     test_loader = _make_loader(X_test, None, batch_size, shuffle=False)
#     emb_test, _ = _get_embeddings(model, test_loader, device)

#     if visualize:
#         viz = Visualizer()
#         # Plot T-SNE of the test embeddings
#         viz.plot_embeddings(emb_test, y_test_raw, title="Verification Embeddings (T-SNE)")
    
#     return _compute_eer(normalize(emb_test), y_test_raw, num_pairs, sampling_mode=sampling_mode)

# # =============================================================================
# # TASK 5: CROSS-SESSION IDENTIFICATION (1:N)
# # =============================================================================
# def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
#     """
#     Train on Session 1 -> Predict Class in Session 2.
    
#     SCENARIO:
#     "I enrolled on Day 1. Can the system recognize me on Day 2?"
    
#     CRITICAL: 
#     We typically filter for 'Overlapping Subjects' (those present in BOTH sessions).
#     If a user is in Session 2 but wasn't in Session 1, the model literally has no 
#     output neuron for them, so we cannot calculate standard accuracy.
#     """
#     _set_seed(seed)
#     device = _get_device(device)
#     print(f"\n[TASK] Cross-Session Identification (Rank-1) on {device}")
    
#     # 1. Find Common Subjects
#     train_subs = np.unique(y_train)
#     test_subs = np.unique(y_test)
#     common_subs = np.intersect1d(train_subs, test_subs)
    
#     if len(common_subs) == 0:
#         raise ValueError("No overlapping subjects between Session 1 and Session 2!")
    
#     print(f"[INFO] Subjects: {len(train_subs)} in Train, {len(test_subs)} in Test.")
#     print(f"[INFO] Evaluated on intersection: {len(common_subs)} common subjects.")
    
#     # 2. Filter Data (Keep only common subjects)
#     mask_train = np.isin(y_train, common_subs)
#     X_train_filt, y_train_filt = x_train[mask_train], y_train[mask_train]
    
#     mask_test = np.isin(y_test, common_subs)
#     X_test_filt, y_test_filt = x_test[mask_test], y_test[mask_test]
    
#     # 3. Remap Labels (Subject ID -> 0..N)
#     # We sort common_subs so the mapping is deterministic (Subj A -> 0, Subj B -> 1...)
#     common_subs.sort()
#     cls_map = {s: i for i, s in enumerate(common_subs)}
    
#     y_train_enc = np.array([cls_map[s] for s in y_train_filt])
#     y_test_enc = np.array([cls_map[s] for s in y_test_filt])
    
#     # 4. Train
#     train_loader = _make_loader(X_train_filt, y_train_enc, batch_size, shuffle=True)
#     test_loader = _make_loader(X_test_filt, y_test_enc, batch_size, shuffle=False)
    
#     in_ch = _detect_channels(X_train_filt)
#     model = model_class(in_channels=in_ch, num_classes=len(common_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training Classifier on Session 1...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     # 5. Test
#     print("[INFO] Phase 2: Predicting on Session 2...")
#     model.eval()
#     all_preds = []
#     all_trues = []
    
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             outputs = model(xb)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_trues.extend(yb.numpy())
            
#     acc = np.mean(np.array(all_preds) == np.array(all_trues))
#     print(f"[RESULT] Cross-Session Identification Accuracy: {acc*100:.2f}%")

#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(all_trues, all_preds, normalize=True)

#     return acc

# # =============================================================================
# # TASK 6: CROSS-SESSION VERIFICATION (1:1)
# # =============================================================================
# def run_cross_session_verification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     """
#     Train on Session 1 -> Verify Identity in Session 2.
    
#     SCENARIO:
#     "I enrolled on Day 1. Can I unlock my phone on Day 2?"
    
#     LOGIC:
#     - Train feature extractor on Session 1.
#     - Extract embeddings for S1 (Templates) and S2 (Probes).
#     - Generate pairs where one element is from S1 and one is from S2.
#     """
#     import collections
#     from scipy.optimize import brentq
#     from scipy.interpolate import interp1d
#     from sklearn.metrics import roc_curve, auc

#     _set_seed(seed)
#     device = _get_device(device)
#     print(f"\n[TASK] Cross-Session Verification (EER) on {device}")
    
#     # 1. Train on Session 1
#     # We use all available S1 data for training to get the best features
#     # y_train_enc, classes = _encode_labels(y_train)
#     all_labels = np.concatenate([y_train, y_test])
#     unique_labels = np.unique(all_labels)
#     label_to_int = {label: i for i, label in enumerate(unique_labels)}
    
#     y_train_int = np.array([label_to_int[l] for l in y_train])
#     y_test_int = np.array([label_to_int[l] for l in y_test])

#     train_loader = _make_loader(x_train, y_train_int, batch_size, shuffle=True)
    
#     in_ch = _detect_channels(x_train)
#     model = model_class(in_channels=in_ch, num_classes=len(label_to_int), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training Feature Extractor on Session 1...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
        
#     # 2. Extract Embeddings
#     print("[INFO] Phase 2: Extracting Embeddings for Pairing...")
#     model.include_top = False # Remove classifier
    
#     # S1 Embeddings (Enrollment)
#     loader1 = _make_loader(x_train, y_train_int, batch_size, shuffle=False)
#     emb1, lab1 = _get_embeddings(model, loader1, device)
    
#     # S2 Embeddings (Probe)
#     loader2 = _make_loader(x_test, y_test_int, batch_size, shuffle=False)
#     emb2, lab2 = _get_embeddings(model, loader2, device)

#     # Normalize
#     emb1 = normalize(emb1)
#     emb2 = normalize(emb2)

#     # 3. Generate Cross-Session Pairs
#     # We cannot use the standard _compute_eer function because that assumes 
#     # a single list of embeddings. Here we must pair (List1 vs List2).
    
#     scores = []
#     labels_pair = []
    
#     # Map indices by subject ID
#     s1_indices = collections.defaultdict(list)
#     for idx, label in enumerate(lab1): s1_indices[label].append(idx)
        
#     s2_indices = collections.defaultdict(list)
#     for idx, label in enumerate(lab2): s2_indices[label].append(idx)
    
#     # Find overlapping subjects for Genuine pairs
#     common_subs = list(set(s1_indices.keys()) & set(s2_indices.keys()))
#     if len(common_subs) < 2:
#         print("[WARN] Not enough overlapping subjects for verification.")
#         return 1.0 # EER = 100% (Fail)

#     print(f"[INFO] Generating {num_pairs} Cross-Session Pairs ({sampling_mode})...")

#     if sampling_mode == "balanced":
#         # A. Genuine Pairs (User X S1 vs User X S2)
#         for _ in range(num_pairs // 2):
#             subj = np.random.choice(common_subs)
#             idx1 = np.random.choice(s1_indices[subj])
#             idx2 = np.random.choice(s2_indices[subj])
#             scores.append(np.dot(emb1[idx1], emb2[idx2]))
#             labels_pair.append(1)
            
#         # B. Imposter Pairs (User X S1 vs User Y S2)
#         all_s1_subs = list(s1_indices.keys())
#         all_s2_subs = list(s2_indices.keys())
#         for _ in range(num_pairs // 2):
#             sub_a = np.random.choice(all_s1_subs)
#             # Pick sub_b such that it is NOT sub_a
#             possible_b = [s for s in all_s2_subs if s != sub_a]
#             if not possible_b: continue # Should not happen if N > 1
            
#             sub_b = np.random.choice(possible_b)
#             idx1 = np.random.choice(s1_indices[sub_a])
#             idx2 = np.random.choice(s2_indices[sub_b])
#             scores.append(np.dot(emb1[idx1], emb2[idx2]))
#             labels_pair.append(0)

#     elif sampling_mode == "random":
#         # Pure random picking
#         indices1 = np.arange(len(lab1))
#         indices2 = np.arange(len(lab2))
#         for _ in range(num_pairs):
#             idx1 = np.random.choice(indices1)
#             idx2 = np.random.choice(indices2)
#             is_same = 1 if lab1[idx1] == lab2[idx2] else 0
#             scores.append(np.dot(emb1[idx1], emb2[idx2]))
#             labels_pair.append(is_same)

#     # 4. Compute Metrics
#     fpr, tpr, _ = roc_curve(labels_pair, scores)
#     roc_auc = auc(fpr, tpr)
#     try:
#         eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#     except:
#         eer = 1.0

#     print(f"[RESULT] Cross-Session EER: {eer:.4f} | AUC: {roc_auc:.4f}")

#     if visualize:
#         viz = Visualizer()
#         # Plot T-SNE of the test embeddings
#         viz.plot_embeddings(emb2, lab2, title="Verification Embeddings (T-SNE)")

#     return {"eer": eer, "auc": roc_auc}

# # run.py
# # ---------------------------------------------------
# # Unified Training & Evaluation Utility for ECG Biometrics
# # ---------------------------------------------------

# import numpy as np
# import random
# from typing import Dict, Any, Union, Callable, Optional, List, Tuple
# import collections

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc, confusion_matrix
# from sklearn.preprocessing import normalize
# from scipy.optimize import brentq
# from scipy.interpolate import interp1d

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader

# from visualizations import Visualizer

# def _set_seed(seed: int = 42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# def _get_device(device: Optional[str] = None) -> str:
#     if device is None or device == "auto":
#         return "cuda" if torch.cuda.is_available() else "cpu"
#     return device

# def _encode_labels(y: np.ndarray):
#     classes, y_enc = np.unique(y, return_inverse=True)
#     return y_enc.astype(np.int64), classes

# def _detect_channels(x: np.ndarray) -> int:
#     if x.ndim == 2: return 1
#     elif x.ndim == 3: return x.shape[1]
#     else: raise ValueError(f"Unexpected input shape: {x.shape}")

# def _make_loader(x, y, batch_size, shuffle=True, device='cpu'):
#     x_t = torch.from_numpy(x).float()
#     if x_t.ndim == 2: x_t = x_t.unsqueeze(1)
#     if y is not None: y_t = torch.from_numpy(y).long(); ds = TensorDataset(x_t, y_t)
#     else: ds = TensorDataset(x_t, torch.zeros(len(x_t)))
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# def _train_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     for xb, yb in loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         logits = model(xb)
#         loss = criterion(logits, yb)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * xb.size(0)
#     return total_loss / len(loader.dataset)

# def _get_embeddings(model, loader, device):
#     model.eval()
#     embeddings, labels = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb = xb.to(device)
#             emb = model(xb)
#             embeddings.append(emb.cpu().numpy())
#             labels.append(yb.numpy())
#     return np.vstack(embeddings), np.concatenate(labels)

# def _compute_eer(embeddings, labels, num_pairs=10000, sampling_mode="balanced"):
#     """
#     Computes EER and AUC using different pair sampling strategies.
#     RETURNS: (eer, auc) tuple
#     """
#     scores = []
#     labels_pair = []
    
#     # -------------------------------------------------------------------------
#     # MODE 1: ALL PAIRS (Similarity Matrix)
#     # -------------------------------------------------------------------------
#     if sampling_mode == "all":
#         print(f"[INFO] Computing ALL pairs (Full Similarity Matrix)...")
#         sim_matrix = np.dot(embeddings, embeddings.T)
#         truth_matrix = (labels[:, None] == labels[None, :]).astype(int)
#         upper_tri_indices = np.triu_indices(len(labels), k=1)
#         scores = sim_matrix[upper_tri_indices]
#         labels_pair = truth_matrix[upper_tri_indices]

#     # -------------------------------------------------------------------------
#     # MODE 2: BALANCED (Standard)
#     # -------------------------------------------------------------------------
#     elif sampling_mode == "balanced":
#         print(f"[INFO] Generating {num_pairs} BALANCED pairs (50/50 split)...")
#         class_indices = collections.defaultdict(list)
#         for idx, label in enumerate(labels):
#             class_indices[label].append(idx)
        
#         valid_genuine_classes = [c for c, idxs in class_indices.items() if len(idxs) >= 2]
        
#         if len(valid_genuine_classes) < 2:
#             print("[WARN] Not enough classes with multiple samples for balanced EER.")
#             return 1.0, 0.5 # Return Tuple

#         # Genuine
#         for _ in range(num_pairs // 2):
#             c = np.random.choice(valid_genuine_classes)
#             idx1, idx2 = np.random.choice(class_indices[c], 2, replace=False)
#             scores.append(np.dot(embeddings[idx1], embeddings[idx2]))
#             labels_pair.append(1)

#         # Imposter
#         all_classes = list(class_indices.keys())
#         for _ in range(num_pairs // 2):
#             c1, c2 = np.random.choice(all_classes, 2, replace=False)
#             idx1 = np.random.choice(class_indices[c1])
#             idx2 = np.random.choice(class_indices[c2])
#             scores.append(np.dot(embeddings[idx1], embeddings[idx2]))
#             labels_pair.append(0)

#     # -------------------------------------------------------------------------
#     # MODE 3: RANDOM
#     # -------------------------------------------------------------------------
#     elif sampling_mode == "random":
#         print(f"[INFO] Generating {num_pairs} RANDOM pairs...")
#         indices = np.arange(len(labels))
#         for _ in range(num_pairs):
#             i1, i2 = np.random.choice(indices, 2, replace=False)
#             scores.append(np.dot(embeddings[i1], embeddings[i2]))
#             labels_pair.append(1 if labels[i1] == labels[i2] else 0)
#     else:
#         raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

#     # Compute Metrics
#     fpr, tpr, thresholds = roc_curve(labels_pair, scores)
#     roc_auc = auc(fpr, tpr)
    
#     try:
#         eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#     except Exception as e:
#         print(f"[WARN] EER calculation failed: {e}")
#         eer = 0.0

#     print(f"[RESULT] Mode: {sampling_mode.upper()} | EER: {eer:.4f} | AUC: {roc_auc:.4f}")
    
#     # FIXED: Return Tuple
#     return eer, roc_auc

# # --- TASK 1: CLOSED-SET IDENTIFICATION ---
# def run_closed_set_identification(x, y, model_class, epochs=30, batch_size=64, lr=1e-3, val_split=0.2, seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Closed-Set Identification on {device}")
#     y_enc, classes = _encode_labels(y)
#     X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=val_split, stratify=y_enc, random_state=seed)
#     train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
#     test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(classes), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs):
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
        
#     model.eval()
#     all_preds, all_trues = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             preds = torch.argmax(model(xb), dim=1)
#             all_preds.append(preds.cpu().numpy()); all_trues.append(yb.cpu().numpy())
#     all_preds = np.concatenate(all_preds); all_trues = np.concatenate(all_trues)
#     acc = np.mean(all_preds == all_trues)
#     print(f"[RESULT] Closed-Set Accuracy: {acc:.4f}")

#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(all_trues, all_preds, normalize=True)

#     return acc

# # --- TASK 2: SUBJECT-DISJOINT IDENTIFICATION (TEMPLATE MATCHING) ---
# def run_subject_disjoint_identification(x, y, model_class, train_subject_ratio=0.7, enrollment_beats=5, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Subject-Disjoint Identification (Template Matching) on {device}")
    
#     # ... (Splitting Logic same as before) ...
#     subjects = np.unique(y)
#     train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
    
#     # Train Feature Extractor (Softmax on Train Subjects)
#     mask_train = np.isin(y, train_subs)
#     X_train, y_train = x[mask_train], y[mask_train]
#     y_train_enc, _ = _encode_labels(y_train)
#     train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(train_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     print("[INFO] Phase 1: Training feature extractor...")
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     # Test (Nearest Neighbor Matching)
#     print("[INFO] Phase 2: Enrollment & Matching (1-NN)...")
#     model.include_top = False
#     mask_test = np.isin(y, test_subs)
#     X_test_all, y_test_all = x[mask_test], y[mask_test]
#     X_enroll, y_enroll, X_query, y_query = [], [], [], []
    
#     for sub in test_subs:
#         sub_beats = X_test_all[y_test_all == sub]
#         if len(sub_beats) <= enrollment_beats: continue
#         X_enroll.append(sub_beats[:enrollment_beats]); y_enroll.extend([sub]*enrollment_beats)
#         X_query.append(sub_beats[enrollment_beats:]); y_query.extend([sub]*(len(sub_beats)-enrollment_beats))
        
#     enroll_loader = _make_loader(np.vstack(X_enroll), None, batch_size, shuffle=False)
#     query_loader = _make_loader(np.vstack(X_query), None, batch_size, shuffle=False)
#     emb_enroll, _ = _get_embeddings(model, enroll_loader, device)
#     emb_query, _ = _get_embeddings(model, query_loader, device)
    
#     # Create Prototypes (Mean of enrollment shots)
#     templates, template_ids = [], []
#     for sub in np.unique(y_enroll):
#         idxs = np.where(np.array(y_enroll) == sub)[0]
#         templates.append(np.mean(emb_enroll[idxs], axis=0))
#         template_ids.append(sub)
    
#     # Cosine Similarity Matching
#     sim_matrix = np.dot(normalize(emb_query), normalize(np.array(templates)).T)
#     preds = np.array(template_ids)[np.argmax(sim_matrix, axis=1)]
#     acc = np.mean(preds == np.array(y_query))
    
#     print(f"[RESULT] Subject-Disjoint Accuracy: {acc:.4f}")
#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(y_query, preds, normalize=True)
#     return acc

# # --- TASK 3: VERIFICATION (RANDOM SPLIT) ---
# def run_verification(x, y, model_class, train_split=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Verification (Closed-Set) on {device}")
#     y_enc, classes = _encode_labels(y)
#     X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=(1-train_split), stratify=y_enc, random_state=seed)
#     train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(classes), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     model.include_top = False
#     test_loader = _make_loader(X_test, y_test, batch_size, shuffle=False)
#     emb_test, labels_test = _get_embeddings(model, test_loader, device)
    
#     if visualize:
#         viz = Visualizer()
#         viz.plot_embeddings(emb_test, labels_test, title="Verification Embeddings (T-SNE)")

#     # Returns (eer, auc)
#     return _compute_eer(normalize(emb_test), labels_test, num_pairs, sampling_mode=sampling_mode)

# # --- TASK 4: SUBJECT-DISJOINT VERIFICATION ---
# def run_subject_disjoint_verification(x, y, model_class, train_subject_ratio=0.7, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Subject-Disjoint Verification on {device}")
#     subjects = np.unique(y)
#     train_subs, test_subs = train_test_split(subjects, train_size=train_subject_ratio, random_state=seed)
    
#     mask_train = np.isin(y, train_subs)
#     X_train, y_train_raw = x[mask_train], y[mask_train]
#     y_train_enc, _ = _encode_labels(y_train_raw)
#     train_loader = _make_loader(X_train, y_train_enc, batch_size, shuffle=True)
#     in_ch = _detect_channels(x)
#     model = model_class(in_channels=in_ch, num_classes=len(train_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     model.include_top = False
#     mask_test = np.isin(y, test_subs)
#     X_test, y_test_raw = x[mask_test], y[mask_test]
#     test_loader = _make_loader(X_test, None, batch_size, shuffle=False)
#     emb_test, _ = _get_embeddings(model, test_loader, device)

#     if visualize:
#         viz = Visualizer()
#         viz.plot_embeddings(emb_test, y_test_raw, title="Verification Embeddings (T-SNE)")
    
#     # Returns (eer, auc)
#     return _compute_eer(normalize(emb_test), y_test_raw, num_pairs, sampling_mode=sampling_mode)

# # --- TASK 5: CROSS-SESSION IDENTIFICATION ---
# def run_cross_session_identification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Cross-Session Identification (Rank-1) on {device}")
    
#     train_subs, test_subs = np.unique(y_train), np.unique(y_test)
#     common_subs = np.intersect1d(train_subs, test_subs)
#     if len(common_subs) == 0: raise ValueError("No overlapping subjects!")
    
#     common_subs.sort()
#     cls_map = {s: i for i, s in enumerate(common_subs)}
    
#     mask_train = np.isin(y_train, common_subs)
#     X_train_filt, y_train_filt = x_train[mask_train], y_train[mask_train]
#     y_train_enc = np.array([cls_map[s] for s in y_train_filt])
    
#     mask_test = np.isin(y_test, common_subs)
#     X_test_filt, y_test_filt = x_test[mask_test], y_test[mask_test]
#     y_test_enc = np.array([cls_map[s] for s in y_test_filt])
    
#     train_loader = _make_loader(X_train_filt, y_train_enc, batch_size, shuffle=True)
#     test_loader = _make_loader(X_test_filt, y_test_enc, batch_size, shuffle=False)
    
#     in_ch = _detect_channels(X_train_filt)
#     model = model_class(in_channels=in_ch, num_classes=len(common_subs), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
    
#     model.eval()
#     all_preds, all_trues = [], []
#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             preds = torch.argmax(model(xb), dim=1)
#             all_preds.extend(preds.cpu().numpy()); all_trues.extend(yb.numpy())
#     acc = np.mean(np.array(all_preds) == np.array(all_trues))
#     print(f"[RESULT] Cross-Session Accuracy: {acc:.4f}")

#     if visualize:
#         viz = Visualizer()
#         viz.plot_confusion_matrix(all_trues, all_preds, normalize=True)
#     return acc

# # --- TASK 6: CROSS-SESSION VERIFICATION ---
# def run_cross_session_verification(x_train, y_train, x_test, y_test, model_class, epochs=30, batch_size=64, lr=1e-3, num_pairs=10000, sampling_mode="balanced", seed=42, device=None, visualize=False):
#     _set_seed(seed); device = _get_device(device)
#     print(f"\n[TASK] Cross-Session Verification on {device}")
    
#     # Train on S1
#     all_labels = np.concatenate([y_train, y_test])
#     unique_labels = np.unique(all_labels)
#     label_to_int = {label: i for i, label in enumerate(unique_labels)}
#     y_train_int = np.array([label_to_int[l] for l in y_train])
#     y_test_int = np.array([label_to_int[l] for l in y_test])

#     train_loader = _make_loader(x_train, y_train_int, batch_size, shuffle=True)
#     in_ch = _detect_channels(x_train)
#     model = model_class(in_channels=in_ch, num_classes=len(label_to_int), include_top=True).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr); criterion = nn.CrossEntropyLoss()
    
#     for ep in range(epochs): 
#         loss = _train_epoch(model, train_loader, optimizer, criterion, device)
#         print(f"    Epoch {ep+1:03d} | Loss: {loss:.4f}")
        
#     model.include_top = False
#     loader1 = _make_loader(x_train, y_train_int, batch_size, shuffle=False)
#     emb1, lab1 = _get_embeddings(model, loader1, device)
#     loader2 = _make_loader(x_test, y_test_int, batch_size, shuffle=False)
#     emb2, lab2 = _get_embeddings(model, loader2, device)

#     # Pairing Logic (Cross-Session)
#     scores, labels_pair = [], []
#     s1_indices = collections.defaultdict(list)
#     for idx, label in enumerate(lab1): s1_indices[label].append(idx)
#     s2_indices = collections.defaultdict(list)
#     for idx, label in enumerate(lab2): s2_indices[label].append(idx)
#     common = list(set(s1_indices.keys()) & set(s2_indices.keys()))
    
#     if len(common) < 2: return 1.0, 0.5

#     emb1 = normalize(emb1); emb2 = normalize(emb2)

#     if sampling_mode == "balanced":
#         # Genuine
#         for _ in range(num_pairs // 2):
#             subj = np.random.choice(common)
#             scores.append(np.dot(emb1[np.random.choice(s1_indices[subj])], emb2[np.random.choice(s2_indices[subj])]))
#             labels_pair.append(1)
#         # Imposter
#         all_s2 = list(s2_indices.keys())
#         for _ in range(num_pairs // 2):
#             s_a = np.random.choice(list(s1_indices.keys()))
#             possible_b = [s for s in all_s2 if s != s_a]
#             if not possible_b: continue
#             s_b = np.random.choice(possible_b)
#             scores.append(np.dot(emb1[np.random.choice(s1_indices[s_a])], emb2[np.random.choice(s2_indices[s_b])]))
#             labels_pair.append(0)
    
#     fpr, tpr, _ = roc_curve(labels_pair, scores)
#     roc_auc = auc(fpr, tpr)
#     try: eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#     except: eer = 1.0

#     print(f"[RESULT] Cross-Session EER: {eer:.4f} | AUC: {roc_auc:.4f}")
#     if visualize:
#         viz = Visualizer()
#         viz.plot_embeddings(emb2, lab2, title="Verification Embeddings (T-SNE)")

#     # FIXED: Return Tuple
#     return eer, roc_auc