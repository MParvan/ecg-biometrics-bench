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

import os
import hashlib
import json

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
    Aggregate enrollment embeddings into subject templates.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix with shape ``(n_samples, feature_dim)``.
    labels : np.ndarray
        Subject labels corresponding to the embeddings.
    method : str
        Template-fusion strategy:

        - ``mean``: arithmetic mean.
        - ``median``: coordinate-wise median.
        - ``trimmed_mean``: coordinate-wise 10% trimmed mean.
        - ``representative``: average of the most centrally consistent
          50% of the selected embeddings, based on cosine similarity.
        - ``soft_centrality``: cosine-centrality-weighted average using
          softmax weighting.
        - ``geometric_median``: iterative geometric median.
        - ``none``: retain all enrollment embeddings without fusion.
    max_beats : int or None
        Maximum number of enrollment embeddings used per subject. The
        first ``max_beats`` samples are selected. ``None`` uses all
        available samples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Template embeddings and their corresponding subject labels.
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
            # Average the most centrally consistent half of the embeddings.
            
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

        elif method == 'soft_centrality':
            # Zero-Shot Self-Attention: Weight beats by their similarity to the group
            norm_embs = normalize(subj_embs, axis=1)
            sim_matrix = cosine_similarity(norm_embs)
            
            # Centrality score: Sum of similarities to all other beats
            scores = np.sum(sim_matrix, axis=1) - 1.0 
            
            # Apply Softmax with Temperature (T) to convert scores to weights
            # T controls sharpness. T=0.1 heavily punishes outliers. T=1.0 is smoother.
            T = 0.1 
            
            # Subtract max for numerical stability before exp
            scores_stable = scores - np.max(scores) 
            exp_scores = np.exp(scores_stable / T)
            weights = exp_scores / np.sum(exp_scores)
            
            # Calculate the weighted average
            template = np.average(subj_embs, axis=0, weights=weights)

        elif method == 'geometric_median':
            # True multi-dimensional median using Weiszfeld's algorithm
            from scipy.spatial.distance import cdist
            
            # Initialize with the standard geometric centroid (mean)
            template = np.mean(subj_embs, axis=0)
            
            # Iterate to converge on the geometric median (usually takes < 10 iterations)
            for _ in range(20):
                distances = cdist(subj_embs, [template])[:, 0]
                # Avoid division by zero for identical points
                distances = np.where(distances == 0, 1e-5, distances)
                
                weights = 1.0 / distances
                weights /= weights.sum()
                
                new_template = np.dot(weights, subj_embs)
                
                # Stop early if converged
                if np.linalg.norm(template - new_template) < 1e-4:
                    template = new_template
                    break
                template = new_template
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

def _run_training_loop(model, train_loader, val_loader, optimizer, criterion, device, epochs, 
                       patience=40, lr_patience=5):
    """
    Advanced Training Loop with LR Decay and Weight Rollback.
    """
    best_val_metric = float('inf')
    epochs_no_improve = 0
    best_model_weights = copy.deepcopy(model.state_dict())
    
    # Advanced LR Scheduler Settings
    lr_factor = 0.8       # Reduce the learning rate
    min_lr = 1e-6         # Do not drop below this

    for ep in range(epochs):
        # --- 1. TRAIN PASS ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- 2. BYPASS FOR NO VALIDATION ---
        # If the user sets val_split=0.0, we just train continuously for all epochs
        if val_loader is None:
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f}")
            best_model_weights = copy.deepcopy(model.state_dict())
            actual_ep = ep + 1
            continue  # Skips the validation and rollback logic!

        # --- 3. VALIDATION PASS ---
        val_metric = float('inf')
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            import torch
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss += criterion(out, yb).item()
            val_metric = val_loss / len(val_loader)
        else:
            # If no validation set, use train loss as the metric to track
            val_metric = train_loss

        # --- 4. CHECK IMPROVEMENT ---
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_metric:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_metric:.4f} | Patience: {epochs_no_improve}/{patience}")

        # --- 5. LEARNING RATE DECAY & ROLLBACK ---
        # If we hit the lr_patience limit, reduce the LR and restore the best weights!
        if epochs_no_improve > 0 and epochs_no_improve % lr_patience == 0:
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr > min_lr:
                new_lr = max(current_lr * lr_factor, min_lr)
                print(f"    -> [LR STEP] No improvement for {lr_patience} epochs.")
                print(f"    -> Rolling back to best weights and reducing LR: {current_lr:.1e} -> {new_lr:.1e}")
                
                # Rollback to best state
                model.load_state_dict(best_model_weights)
                
                # Apply new LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

        # --- 6. HARD EARLY STOPPING ---
        if epochs_no_improve >= patience:
            print(f"\n[INFO] Hard early stopping triggered after {ep+1} epochs.")
            break

    # Restore the absolute best weights before returning the model
    print(f"[INFO] Training complete. Restoring best model weights.")
    model.load_state_dict(best_model_weights)
    model.actual_epochs = ep + 1
    
    return model

def _run_train_loop_unseen_subjects(model, train_loader, val_loader_seen, val_loader_unseen, optimizer, criterion, device, 
                                    epochs, matching_method='cosine', patience=40, lr_patience=15):
    """
    Advanced Training Loop for Open-Set Tasks using a Composite Metric.
    Score = (Normalized Seen Val Loss) + (Unseen Val EER)
    """
    best_combined_score = float('inf')
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())
    
    lr_factor = 0.8
    min_lr = 1e-6
    actual_ep = epochs
    baseline_seen_loss = None # Used to normalize the CE Loss
    
    for ep in range(epochs):
        # --- 1. TRAIN PASS ---
        model.train()
        model.include_top = True
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)

        # --- 2. BYPASS FOR NO VALIDATION ---
        # If the user sets val_split=0.0, we just train continuously for all epochs
        if val_loader_seen is None and val_loader_unseen is None:
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f}")
            best_model_state = copy.deepcopy(model.state_dict())
            actual_ep = ep + 1
            continue  # Skips the rest of the loop!
        
        # --- 3. VALIDATION PASS: SEEN SUBJECTS (Cross-Entropy Loss) ---
        model.eval()
        val_loss_seen = 0.0
        if val_loader_seen is not None:
            with torch.no_grad():
                for xb, yb in val_loader_seen:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss_seen += criterion(out, yb).item()
            val_loss_seen /= len(val_loader_seen)
        else:
            val_loss_seen = train_loss # Fallback
            
        # Normalize the loss so it scales similarly to EER (1.0 -> 0.0)
        if baseline_seen_loss is None:
            baseline_seen_loss = val_loss_seen + 1e-8
        norm_val_loss = val_loss_seen / baseline_seen_loss
        
        # --- 4. VALIDATION PASS: UNSEEN SUBJECTS (EER) ---
        val_eer_unseen = 1.0
        if val_loader_unseen is not None:
            model.include_top = False # Extract Features
            
            # Handle Cross-Session (Tuple) vs Intra-Session (Single)
            if isinstance(val_loader_unseen, tuple):
                val_s1, val_s2 = val_loader_unseen
                val_emb_s1, val_lab_s1 = _get_embeddings(model, val_s1, device)
                val_emb_s2, val_lab_s2 = _get_embeddings(model, val_s2, device)
                val_scores, val_pairs = _generate_pairs(val_emb_s2, val_lab_s2, val_emb_s1, val_lab_s1, 
                                                        2000, "balanced", matching_method)
            else:
                val_emb, val_lab = _get_embeddings(model, val_loader_unseen, device)
                val_scores, val_pairs = _generate_pairs(val_emb, val_lab, None, None, 
                                                        2000, "balanced", matching_method)
            
            if len(val_pairs) > 0:
                fpr, tpr, _ = roc_curve(val_pairs, val_scores)
                try: val_eer_unseen = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                except: pass
                
        # --- 5. COMPOSITE SCORE CALCULATION ---
        combined_score = norm_val_loss + val_eer_unseen
        
        # --- 6. CHECK IMPROVEMENT & ROLLBACK ---
        if combined_score < best_combined_score:
            best_combined_score = combined_score
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f} | Seen Val Loss: {val_loss_seen:.4f} | Unseen EER: {val_eer_unseen:.4f} | Score: {combined_score:.4f}")
        else:
            patience_counter += 1
            print(f"Epoch {ep+1}/{epochs} | Train Loss: {train_loss:.4f} | Seen Val Loss: {val_loss_seen:.4f} | Unseen EER: {val_eer_unseen:.4f} | Score: {combined_score:.4f} | Patience: {patience_counter}/{patience}")
            
            if patience_counter > 0 and patience_counter % lr_patience == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr > min_lr:
                    new_lr = max(current_lr * lr_factor, min_lr)
                    print(f"    -> [LR STEP] No improvement for {lr_patience} epochs.")
                    print(f"    -> Rolling back to best weights and reducing LR: {current_lr:.1e} -> {new_lr:.1e}")
                    model.load_state_dict(best_model_state)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
            
            if patience_counter >= patience:
                print(f"--> Hard early stopping triggered at epoch {ep+1}.")
                actual_ep = ep + 1
                break
                
    print(f"[INFO] Training complete. Restoring best model weights.")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.actual_epochs = actual_ep

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

def _summarize_verification_pairs(
    labels_pair,
    target_far=0.001,
):
    """
    Summarize the comparison counts supporting verification metrics.

    The minimum nonzero empirical FAR is one false acceptance divided by
    the number of impostor comparisons. A target FAR is empirically
    resolvable when this minimum step is no larger than the target.
    """
    labels_pair = np.asarray(labels_pair)

    if labels_pair.ndim != 1:
        raise ValueError(
            "labels_pair must be a one-dimensional array."
        )

    if len(labels_pair) == 0:
        raise ValueError(
            "labels_pair cannot be empty."
        )

    unique_labels = set(np.unique(labels_pair).tolist())

    if not unique_labels.issubset({0, 1, False, True}):
        raise ValueError(
            "labels_pair must contain only binary labels 0 and 1."
        )

    try:
        target_far = float(target_far)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "target_far must be numeric."
        ) from error

    if not 0.0 < target_far < 1.0:
        raise ValueError(
            "target_far must satisfy 0 < target_far < 1."
        )

    genuine_count = int(np.sum(labels_pair == 1))
    impostor_count = int(np.sum(labels_pair == 0))
    total_count = int(len(labels_pair))

    minimum_nonzero_far = (
        1.0 / impostor_count
        if impostor_count > 0
        else None
    )

    target_resolvable = (
        minimum_nonzero_far is not None
        and minimum_nonzero_far <= target_far
    )

    return {
        "Total Verification Pairs": total_count,
        "Genuine Pairs": genuine_count,
        "Impostor Pairs": impostor_count,
        "Target FAR": target_far,
        "Minimum Non-Zero Empirical FAR": minimum_nonzero_far,
        "Target FAR Empirically Resolvable": target_resolvable,
    }

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
                    num_pairs=10000, sampling_mode="all", matching_method='cosine'):
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

        # Subjects eligible for genuine comparisons.
        # In intra-set evaluation, each subject must have at least two
        # different samples so that self-pairs cannot be generated.
        if match_two_sets:
            genuine_subjects = common_subs
        else:
            genuine_subjects = [
                subject
                for subject in common_subs
                if len(s1_idx[subject]) >= 2
            ]

        if not genuine_subjects:
            return np.array([]), np.array([])

        # Genuine pairs
        for _ in range(num_pairs // 2):
            subj = np.random.choice(genuine_subjects)

            if match_two_sets:
                # Probe versus enrollment/template.
                idx1 = np.random.choice(s1_idx[subj])
                idx2 = np.random.choice(s2_idx[subj])
            else:
                # Intra-set evaluation requires two distinct samples.
                candidate_indices = np.asarray(s1_idx[subj])
                idx1, idx2 = np.random.choice(
                    candidate_indices,
                    size=2,
                    replace=False,
                )

            score = _compute_pair_score(
                embeddings1[idx1],
                embeddings2[idx2],
                method=matching_method,
            )
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

def _apply_outlier_filter(
    x,
    y,
    sqi_scores,
    absolute_threshold=0.05,
    keep_percentage=0.8,
    apply_subject_ranking=True,
):
    """
    Filter ECG samples using a Signal Quality Index.

    The absolute threshold is identity-independent and can be used for
    training, enrollment, validation, and probe data.

    When ``apply_subject_ranking=True``, the function additionally retains
    only the highest-quality percentage of samples separately for each
    subject. This option is appropriate for controlled training or
    enrollment data, but must not be used for operational probe data because
    it requires access to ground-truth subject labels.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sqi_scores = np.asarray(sqi_scores)

    if len(x) != len(y):
        raise ValueError(
            "[ERROR] x and y must contain the same number of samples."
        )

    if len(sqi_scores) != len(x):
        raise ValueError(
            "[ERROR] sqi_scores must contain one score per sample."
        )

    if not 0.0 <= absolute_threshold <= 1.0:
        raise ValueError(
            "[ERROR] absolute_threshold must be between 0.0 and 1.0."
        )

    if not 0.0 < keep_percentage <= 1.0:
        raise ValueError(
            "[ERROR] keep_percentage must be greater than 0.0 "
            "and less than or equal to 1.0."
        )

    # Stage 1: identity-independent absolute SQI threshold.
    threshold_mask = sqi_scores >= absolute_threshold
    surviving_indices = np.flatnonzero(threshold_mask)

    if not apply_subject_ranking:
        final_indices = surviving_indices
    else:
        # Stage 2: controlled per-subject ranking for training/enrollment.
        surviving_labels = y[surviving_indices]
        surviving_scores = sqi_scores[surviving_indices]

        selected_indices = []

        for subject in np.unique(surviving_labels):
            local_subject_indices = np.flatnonzero(
                surviving_labels == subject
            )

            number_to_keep = max(
                1,
                int(len(local_subject_indices) * keep_percentage),
            )

            subject_scores = surviving_scores[local_subject_indices]

            ranked_local_indices = local_subject_indices[
                np.argsort(subject_scores)[::-1]
            ]

            selected_indices.extend(
                surviving_indices[
                    ranked_local_indices[:number_to_keep]
                ]
            )

        final_indices = np.asarray(
            sorted(selected_indices),
            dtype=int,
        )

    print(
        "[INFO] Outlier Filter: "
        f"Dropped {len(x) - len(final_indices)} samples. "
        f"Retained {len(final_indices)}."
    )

    return x[final_indices], y[final_indices]


def _generate_config_hash(config_dict):
    """Creates a deterministic short hash from a dictionary of parameters."""
    # Convert dict to a sorted JSON string to ensure consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:12]

class CacheManager:
    def __init__(self, base_dir="precomputed"):
        self.data_dir = os.path.join(base_dir, "data")
        self.weight_dir = os.path.join(base_dir, "weights")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)

    def get_data_cache(self, config_dict):
        uid = _generate_config_hash(config_dict)
        data_path = os.path.join(self.data_dir, f"{uid}.npz")
        if os.path.exists(data_path):
            data = np.load(data_path, allow_pickle=True)
            return {k: data[k] for k in data.files}, uid
        return None, uid

    def save_data_cache(self, arrays_dict, config_dict, uid):
        np.savez_compressed(os.path.join(self.data_dir, f"{uid}.npz"), **arrays_dict)
        with open(os.path.join(self.data_dir, f"{uid}.json"), "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

    def get_weight_cache(self, config_dict, model, device):
        uid = _generate_config_hash(config_dict)
        weight_path = os.path.join(self.weight_dir, f"{uid}.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            return model, uid
        return None, uid

    def save_weight_cache(self, model, config_dict, uid):
        torch.save(model.state_dict(), os.path.join(self.weight_dir, f"{uid}.pth"))
        with open(os.path.join(self.weight_dir, f"{uid}.json"), "w") as f:
            json.dump(config_dict, f, indent=4, default=str)

