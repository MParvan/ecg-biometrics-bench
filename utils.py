# utils.py

import numpy as np
import random
import collections
import copy
from typing import Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.optimize import brentq
from scipy.stats import trim_mean, kurtosis
from scipy.spatial.distance import cdist, correlation
from scipy.interpolate import interp1d

import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from scipy.stats import kurtosis
from sklearn.metrics import roc_curve, auc

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

def _create_templates(embeddings, labels, method='mean', max_beats=None):
    """
    Aggregates embeddings into a SINGLE template vector per class.
    
    Args:
        embeddings: (N, D) array.
        labels: (N,) array.
        method: 'mean', 'median', 'trimmed_mean', 'representative'.
        max_beats: If set (e.g. 5), uses only the FIRST 5 beats available for the subject.
    """
    if method == 'none' or method is None:
        return embeddings, labels
    
    if max_beats is not None:
        try: max_beats = int(max_beats)
        except: max_beats = None

    unique_labels = np.unique(labels)
    templates = []
    new_labels = []
    
    feature_dim = embeddings.shape[1]
    
    for label in unique_labels:
        # 1. Get all available indices for this subject
        idxs = np.where(labels == label)[0]
        
        # 2. Apply Input Constraint (First N Beats)
        if max_beats is not None and len(idxs) > max_beats:
            idxs = idxs[:max_beats]
            
        subj_embs = embeddings[idxs]
        if len(subj_embs) == 0: continue

        # 3. Apply Fusion Method on the selected subset
        if method == 'representative':
            # Logic: Calculate similarity among these beats, pick best half, average them.
            
            # A. Calculate Similarity Matrix
            norm_embs = normalize(subj_embs, axis=1)
            sim_matrix = cosine_similarity(norm_embs)
            
            # B. Score = Sum of similarities (minus self)
            scores = np.sum(sim_matrix, axis=1) - 1.0
            
            # C. Determine K (Best 50% of the selected subset)
            # Minimum 1 beat, otherwise half the set
            K = max(1, len(subj_embs) // 2)
            
            # D. Select Top K indices within this subset
            top_k_indices = np.argsort(scores)[-K:]
            
            # E. Average
            template = np.mean(subj_embs[top_k_indices], axis=0)

        elif method == 'mean':
            template = np.mean(subj_embs, axis=0)
            
        elif method == 'median':
            template = np.median(subj_embs, axis=0)
            
        elif method == 'trimmed_mean':
            if len(subj_embs) < 5: 
                template = np.mean(subj_embs, axis=0)
            else: 
                template = trim_mean(subj_embs, proportiontocut=0.1, axis=0)
        else:
            raise ValueError(f"Unknown template method: {method}")
        
        # Ensure shape consistency
        if template.shape != (feature_dim,):
            template = template.reshape(feature_dim)
            
        templates.append(template)
        new_labels.append(label)
        
    return np.stack(templates), np.array(new_labels)

# =============================================================================
# 2. MATCHING & SCORE HELPERS (NEW SECTION)
# =============================================================================

def _compute_score_matrix(probes, gallery, method='cosine'):
    """
    Computes similarity matrix for Identification (1:N).
    Returns matrix of shape (N_probes, N_gallery).
    Higher score = Better match.
    """
    if method == 'cosine':
        # Cosine Similarity: Dot product of normalized vectors
        p_norm = normalize(probes, axis=1)
        g_norm = normalize(gallery, axis=1)
        return np.dot(p_norm, g_norm.T)
    
    elif method == 'euclidean':
        # Euclidean Distance: Lower is better. Convert to negative distance.
        dists = cdist(probes, gallery, metric='euclidean')
        return -dists
    
    elif method == 'manhattan':
        # L1 Distance (Cityblock)
        dists = cdist(probes, gallery, metric='cityblock')
        return -dists
    
    elif method == 'correlation':
        # Pearson Correlation. 
        # Scipy cdist 'correlation' returns (1 - correlation).
        # We want correlation, so we return (1 - dist)
        dists = cdist(probes, gallery, metric='correlation')
        return 1.0 - dists
    
    else:
        raise ValueError(f"Unknown matching method: {method}")

def _compute_pair_score(v1, v2, method='cosine'):
    """
    Computes similarity score for a single pair (1:1) for Verification.
    """
    if method == 'cosine':
        # Assume v1, v2 are raw. Normalize first.
        n1 = v1 / (np.linalg.norm(v1) + 1e-10)
        n2 = v2 / (np.linalg.norm(v2) + 1e-10)
        return np.dot(n1, n2)
    
    elif method == 'euclidean':
        dist = np.linalg.norm(v1 - v2)
        return -dist
    
    elif method == 'manhattan':
        dist = np.sum(np.abs(v1 - v2))
        return -dist

    elif method == 'correlation':
        # 1 - correlation_dist = pearson_corr
        dist = correlation(v1, v2)
        return 1.0 - dist
    
    else:
        raise ValueError(f"Unknown matching method: {method}")
    
def _apply_score_fusion(scores, labels, fusion_size=1):
    """
    Applies Score-Level Fusion (Continuous Authentication) to probe beats.
    Groups a subject's test scores into blocks of 'fusion_size' and averages them.
    """
    if fusion_size <= 1:
        return scores, labels

    print(f"[INFO] Applying Score-Level Fusion (Averaging every {fusion_size} consecutive probe beats)...")
    fused_scores, fused_labels = [], []
    
    for subj in np.unique(labels):
        subj_scores = scores[labels == subj]
        num_blocks = len(subj_scores) // fusion_size
        
        # If a subject has fewer beats than the fusion size, just average what they have
        if num_blocks == 0:
            fused_scores.append(np.mean(subj_scores, axis=0))
            fused_labels.append(subj)
            continue
            
        # Extract non-overlapping blocks and average them
        for i in range(num_blocks):
            block = subj_scores[i * fusion_size : (i + 1) * fusion_size]
            fused_scores.append(np.mean(block, axis=0))
            fused_labels.append(subj)
            
    return np.array(fused_scores), np.array(fused_labels)

def _find_optimal_threshold(scores, labels):
    """
    Finds the global threshold where FAR and FRR intersect (EER) 
    using a calibration (Validation/Train) dataset.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    optimal_idx = np.nanargmin(np.absolute(fnr - fpr))
    return thresholds[optimal_idx]

def _evaluate_with_global_threshold(test_scores, test_labels, global_threshold):
    """
    Applies the frozen global threshold to the Test set to calculate 
    real-world deployment metrics.
    """
    predictions = (test_scores >= global_threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (test_labels == 1))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))
    
    accuracy = (tp + tn) / len(test_labels) if len(test_labels) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"\n[REAL-WORLD CALIBRATION]")
    print(f"  > Applied Global Threshold : {global_threshold:.4f}")
    print(f"  > Deployed Accuracy        : {accuracy:.2%}")
    print(f"  > Deployed FAR             : {far:.2%}")
    print(f"  > Deployed FRR             : {frr:.2%}")
    
    return accuracy, far, frr

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

def _validate_epoch(model, loader, criterion, device):
    """Calculates loss on validation set without backprop."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience=10):
    """
    Runs training with optional validation and early stopping.
    If val_loader is provided:
      - Tracks validation loss.
      - Stops if no improvement for 'patience' epochs.
      - Restores the best model weights before returning.
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for ep in range(epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        
        log_msg = f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f}"
        
        if val_loader is not None:
            val_loss = _validate_epoch(model, val_loader, criterion, device)
            log_msg += f" | Val Loss: {val_loss:.4f}"
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0 # Reset
            else:
                patience_counter += 1
                
            print(log_msg)
            
            # Early Stopping Check
            if patience_counter >= patience:
                print(f"--> Early stopping triggered at epoch {ep+1}. No improvement for {patience} epochs.")
                break
        else:
            print(log_msg)

    # Restore best model if validation was used
    if best_model_state is not None:
        print(f"--> Restoring best model from epoch with Val Loss: {best_val_loss:.4f}")
        model.load_state_dict(best_model_state)

    model.actual_epochs = ep + 1
    
    return model

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
def _generate_pairs(embeddings1, labels1, embeddings2=None, labels2=None, 
                    num_pairs=10000, sampling_mode="balanced", matching_method='cosine'):
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
    
    # RENAMED FOR CLARITY: 
    # If True, we match Set 1 (Probes) against Set 2 (Templates).
    # If False, we match Set 1 against itself (Intra-set evaluation).
    match_two_sets = (embeddings2 is not None)
    
    if not match_two_sets:
        embeddings2 = embeddings1
        labels2 = labels1

    if len(labels1) == 0 or len(labels2) == 0:
        return np.array([]), np.array([])

    # --------------------------------------------------------
    # MODE A: ALL PAIRS (Full Matrix)
    # --------------------------------------------------------
    if sampling_mode == "all":
        print(f"[INFO] generating ALL pairs (Full Matrix evaluation)...")
        sim_matrix = _compute_score_matrix(embeddings1, embeddings2, method=matching_method)
        truth_matrix = (labels1[:, None] == labels2[None, :]).astype(int)
        
        if match_two_sets:
            scores = sim_matrix.flatten()
            labels_pair = truth_matrix.flatten()
        else:
            upper_tri = np.triu_indices(len(labels1), k=1)
            scores = sim_matrix[upper_tri]
            labels_pair = truth_matrix[upper_tri]
            
    # --------------------------------------------------------
    # MODE B: BALANCED (Standard)
    # --------------------------------------------------------
    elif sampling_mode == "balanced":
        s1_idx = collections.defaultdict(list)
        s2_idx = collections.defaultdict(list)
        for i, l in enumerate(labels1): s1_idx[l].append(i)
        for i, l in enumerate(labels2): s2_idx[l].append(i)
        
        common_subs = list(set(s1_idx.keys()) & set(s2_idx.keys()))
        if len(common_subs) < 2: return np.array([]), np.array([])

        # Genuine Pairs
        for _ in range(num_pairs // 2):
            subj = np.random.choice(common_subs)
            idx1 = np.random.choice(s1_idx[subj])
            
            if match_two_sets:
                # E.g., Probe vs Template
                idx2 = np.random.choice(s2_idx[subj])
            else:
                # E.g., Test Beat vs Test Beat (Must ensure they are different beats)
                if len(s1_idx[subj]) < 2: continue
                idx2 = np.random.choice(s1_idx[subj], 2, replace=False)[1]
                
            score = _compute_pair_score(embeddings1[idx1], embeddings2[idx2], method=matching_method)
            scores.append(score)
            labels_pair.append(1)

        # Imposter Pairs
        all_s1 = list(s1_idx.keys())
        all_s2 = list(s2_idx.keys())
        for _ in range(num_pairs // 2):
            s_a = np.random.choice(all_s1)
            possible_b = [s for s in all_s2 if s != s_a]
            if not possible_b: continue
            s_b = np.random.choice(possible_b)
            
            idx1 = np.random.choice(s1_idx[s_a])
            idx2 = np.random.choice(s2_idx[s_b])
            
            score = _compute_pair_score(embeddings1[idx1], embeddings2[idx2], method=matching_method)
            scores.append(score)
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
            
            if not match_two_sets and i1 == i2: continue
            
            score = _compute_pair_score(embeddings1[i1], embeddings2[i2], method=matching_method)
            scores.append(score)
            labels_pair.append(1 if labels1[i1] == labels2[i2] else 0)
            
    return np.array(scores), np.array(labels_pair)

# =============================================================================
# HELPER: PREPROCESSING & OUTLIER FILTERING
# =============================================================================
def calculate_kurtosis_sqi(x):
    """
    Calculates a simple Signal Quality Index (SQI) based on Kurtosis.
    Clean ECGs have high kurtosis (sharp R-peaks). Noisy ECGs have low kurtosis.
    
    Args:
        x (np.ndarray): Array of ECG beats, shape (N_beats, signal_length)
    Returns:
        sqi_scores (np.ndarray): Array of quality scores from 0.0 to 1.0
    """
    print("[INFO] Calculating SQI scores based on signal Kurtosis...")
    
    # 1. Calculate kurtosis for each beat across its signal length (axis=1)
    # Fisher=False gives Pearson's kurtosis (normal distribution = 3.0)
    raw_kurt = kurtosis(x, axis=1, fisher=False)
    
    # 2. Handle any potential NaNs from dead signals (e.g., all zeros)
    raw_kurt = np.nan_to_num(raw_kurt, nan=0.0)
    
    # 3. Normalize to a 0.0 - 1.0 scale. 
    # (Typically, ECG kurtosis ranges from 3 up to 20+. We clip at 15 to handle extreme spikes).
    min_kurt = 3.0  # Gaussian noise baseline
    max_kurt = 15.0 # Very sharp, clean R-peak
    
    # Clip and normalize
    sqi_scores = np.clip((raw_kurt - min_kurt) / (max_kurt - min_kurt), 0.0, 1.0)
    
    return sqi_scores

def _compute_sqi(x, method):
    """
    Central dispatcher for calculating Signal Quality Index (SQI).
    Easily expandable for future methods.
    """
    if method == 'kurtosis':
        return calculate_kurtosis_sqi(x) # The function we wrote earlier
    elif method == 'snr':
        # Placeholder for a future Signal-to-Noise Ratio method
        raise NotImplementedError("SNR method is not yet implemented.")
    elif method == 'template_matching':
        # Placeholder for another future method
        raise NotImplementedError("Template matching SQI is not yet implemented.")
    else:
        raise ValueError(f"[ERROR] Unknown SQI calculation method: '{method}'")

def _apply_outlier_filter(x, y, sqi_scores, absolute_threshold=0.05, keep_percentage=0.8):
    """
    Filters out noisy ECG beats based on Signal Quality Index (SQI).
    """
    if sqi_scores is None or len(sqi_scores) != len(x):
        raise ValueError("[ERROR] To use outlier_filtering=True, you must provide an 'sqi_scores' array of the same length as 'x'.")

    # Stage 1: Absolute Threshold
    stage1_mask = sqi_scores >= absolute_threshold
    x_stg1, y_stg1, sqi_stg1 = x[stage1_mask], y[stage1_mask], sqi_scores[stage1_mask]
    original_indices = np.where(stage1_mask)[0]
    
    # Stage 2: Per-Subject Percentage
    final_indices = []
    for subject in np.unique(y_stg1):
        subject_idx = np.where(y_stg1 == subject)[0]
        num_to_keep = max(1, int(len(subject_idx) * keep_percentage))
        
        subj_sqi = sqi_stg1[subject_idx]
        sorted_subj_idx = subject_idx[np.argsort(subj_sqi)[::-1]] # Highest SQI first
        
        final_indices.extend(original_indices[sorted_subj_idx[:num_to_keep]])

    final_indices = sorted(final_indices) # Maintain chronological order
    
    print(f"[INFO] Outlier Filter: Dropped {len(x) - len(final_indices)} noisy beats. Retained {len(final_indices)}.")
    return x[final_indices], y[final_indices]
