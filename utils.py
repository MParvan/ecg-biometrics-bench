# utils.py

import numpy as np
import random
import collections
import copy
from typing import Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import trim_mean, kurtosis
from scipy.spatial.distance import cdist, correlation

import pickle
import tempfile
import zipfile

import os
import hashlib
import json

import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
from sklearn.metrics import (
    auc,
    det_curve,
    roc_curve,
)

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

def _create_templates(embeddings, labels, method='mean', max_beats=None, provenance=None):
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
        Enrollment budget: the first ``max_beats`` beats per subject are used,
        selected in genuine source order when ``provenance`` is supplied and in
        input order otherwise. ``None`` uses all available beats.
    provenance : BeatProvenance or None
        Optional per-beat source provenance, index-aligned with ``embeddings``.
        When supplied for a fusing method it orders the finite first-N budget by
        source. ``method='none'`` returns the enrollment embeddings before the
        budget is applied; its finite-budget semantics are handled separately.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Template embeddings and their corresponding subject labels.
    """
    if method == 'none' or method is None:
        return embeddings, labels

    if provenance is not None:
        provenance.validate(len(labels))
    
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
        
        # 2. Apply the enrolment budget: the genuinely first N beats in source
        # order when provenance is available, otherwise input order.
        if max_beats is not None and len(idxs) > max_beats:
            if provenance is not None:
                order = _source_order_indices(provenance.subset(idxs))
                idxs = idxs[order][:max_beats]
            else:
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
    
def _source_order_indices(provenance):
    """
    Order the beats of one subject and role by truthful source order.

    When every acquisition that needs ordering states a genuine
    ``acquisition_time`` the beats are ordered chronologically; otherwise the
    deterministic ``acquisition_order`` orders the whole set (dated and undated
    are never mixed and called chronology). Within an acquisition the order is
    ``source_segment_order`` then the within-source position (``rpeak_index``
    when present, otherwise ``beat_ordinal``). The original index breaks ties so
    the order is fully deterministic.
    """
    columns = provenance.columns
    count = len(provenance)
    if count == 0:
        return np.empty((0,), dtype=int)

    times = columns["acquisition_time"]
    all_dated = all(value is not None for value in times)
    acquisition_order = [int(value) for value in columns["acquisition_order"]]
    if all_dated:
        # Genuine time first; the stable acquisition_order breaks a tie between
        # records sharing a date so shuffled array position never decides it.
        primary = list(times)
    else:
        primary = list(acquisition_order)

    record_id = [str(value) for value in columns["record_id"]]
    segment_id = [str(value) for value in columns["source_segment_id"]]
    rpeak = np.asarray(columns["rpeak_index"])
    use_rpeak = bool(rpeak.size and np.all(rpeak >= 0))
    segment_order = columns["source_segment_order"]
    beat_ordinal = columns["beat_ordinal"]

    def sort_key(index):
        within = (
            float(rpeak[index]) if use_rpeak else int(beat_ordinal[index])
        )
        return (
            primary[index],
            acquisition_order[index],
            record_id[index],
            float(segment_order[index]),
            segment_id[index],
            within,
            index,
        )

    return np.asarray(sorted(range(count), key=sort_key), dtype=int)


def _source_block_keys(provenance):
    """
    The source block each beat belongs to. A fused probe group never bridges the
    sessions/acquisitions, physical records or source segments of a subject.
    """
    columns = provenance.columns
    return list(
        zip(
            columns["session_id"].tolist(),
            columns["record_id"].tolist(),
            columns["source_segment_id"].tolist(),
        )
    )


def _empty_fusion_diagnostics(count):
    return {
        "fusion_size": 1,
        "raw_probe_observations": int(count),
        "fused_probe_decisions": int(count),
        "dropped_remainder_observations": 0,
        "source_blocks_below_fusion_size": 0,
        "identities_without_a_fused_decision": 0,
    }


def _apply_score_fusion(
    scores,
    labels,
    fusion_size=1,
    provenance=None,
    return_diagnostics=False,
):
    """
    Fuse the probe scores of one subject in fixed-depth groups within a source
    block.

    A fused group averages exactly ``fusion_size`` probe observations that share
    a subject, session/acquisition, physical record and source segment, taken in
    truthful source order. Groups never bridge those boundaries. The final
    incomplete remainder of a block (fewer than ``fusion_size`` observations) is
    dropped rather than scored at a smaller depth, and a block holding fewer than
    ``fusion_size`` observations produces no fused decision.

    ``fusion_size = 1`` returns the scores and labels unchanged (one-beat
    scoring). For ``fusion_size > 1`` per-beat provenance is required so the
    source boundaries can be honoured.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if fusion_size <= 1:
        if return_diagnostics:
            return scores, labels, _empty_fusion_diagnostics(len(labels))
        return scores, labels

    if provenance is None:
        raise ValueError(
            "Probe fusion with fusion_size > 1 requires per-beat provenance so "
            "fused groups stay within one subject, acquisition, record and "
            "source segment. Provide provenance or evaluate with fusion_size = 1."
        )
    provenance.validate(len(labels))

    print(
        f"[INFO] Applying score-level fusion in groups of {fusion_size} within "
        "one source segment at a time..."
    )

    fused_scores = []
    fused_labels = []
    dropped_remainder = 0
    short_blocks = 0
    zero_decision_subjects = 0

    block_keys = _source_block_keys(provenance)

    for subject in np.unique(labels):
        subject_indices = np.flatnonzero(labels == subject)
        subject_produced = 0

        blocks = {}
        for index in subject_indices:
            blocks.setdefault(block_keys[index], []).append(index)

        for block_key in sorted(blocks):
            member_indices = np.asarray(blocks[block_key], dtype=int)
            order = _source_order_indices(provenance.subset(member_indices))
            ordered = member_indices[order]

            complete_groups = len(ordered) // fusion_size
            if complete_groups == 0:
                short_blocks += 1
                dropped_remainder += len(ordered)
                continue

            for group in range(complete_groups):
                members = ordered[
                    group * fusion_size : (group + 1) * fusion_size
                ]
                fused_scores.append(np.mean(scores[members], axis=0))
                fused_labels.append(subject)
                subject_produced += 1

            dropped_remainder += len(ordered) - complete_groups * fusion_size

        if subject_produced == 0:
            zero_decision_subjects += 1

    if fused_scores:
        fused_scores = np.asarray(fused_scores)
    else:
        fused_scores = np.empty((0,) + scores.shape[1:], dtype=scores.dtype)
    fused_labels = np.asarray(fused_labels)

    if return_diagnostics:
        diagnostics = {
            "fusion_size": int(fusion_size),
            "raw_probe_observations": int(len(labels)),
            "fused_probe_decisions": int(len(fused_labels)),
            "dropped_remainder_observations": int(dropped_remainder),
            "source_blocks_below_fusion_size": int(short_blocks),
            "identities_without_a_fused_decision": int(zero_decision_subjects),
        }
        return fused_scores, fused_labels, diagnostics

    return fused_scores, fused_labels


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
    Advanced Training Loop for Subject-Disjoint Tasks using a Composite Metric.
    Score = (Normalized Seen Val Loss) + (Subject-Disjoint Validation EER)
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
            
            if len(val_pairs) == 0:
                raise ValueError(
                    "Unseen-subject validation produced no "
                    "verification pairs."
                )

            (
                validation_scores,
                validation_labels,
            ) = _validate_verification_curve_inputs(
                val_scores,
                val_pairs,
            )

            (
                validation_false_accept_rates,
                validation_true_accept_rates,
                _,
            ) = roc_curve(
                validation_labels,
                validation_scores,
                drop_intermediate=False,
            )

            val_eer_unseen = (
                _interpolate_equal_error_rate(
                    validation_false_accept_rates,
                    (
                        1.0
                        - validation_true_accept_rates
                    ),
                )
            )
                
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
_DEFAULT_VERIFICATION_TARGET_FARS = (
    0.1,
    0.01,
    0.001,
    0.0001,
)


def _finite_float_or_none(
    value,
):
    """
    Return a finite Python float or None.

    ROC threshold arrays can contain positive infinity for the
    reject-all operating point, which strict JSON cannot represent.
    """
    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def _normalize_target_fars(
    target_fars,
):
    """
    Validate and normalize requested verification FAR operating points.
    """
    if target_fars is None:
        target_fars = (
            _DEFAULT_VERIFICATION_TARGET_FARS
        )

    try:
        target_fars = list(
            target_fars
        )
    except TypeError as error:
        raise ValueError(
            "target_fars must be an iterable "
            "of numeric FAR values."
        ) from error

    if not target_fars:
        raise ValueError(
            "target_fars cannot be empty."
        )

    normalized_fars = []

    for target_far in target_fars:
        if isinstance(
            target_far,
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError(
                "FAR targets must be numeric, "
                "not Boolean."
            )

        try:
            target_far = float(
                target_far
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Every FAR target must be numeric."
            ) from error

        if not np.isfinite(
            target_far
        ):
            raise ValueError(
                "Every FAR target must be finite."
            )

        if not 0.0 < target_far < 1.0:
            raise ValueError(
                "Every FAR target must satisfy "
                "0 < FAR < 1."
            )

        normalized_fars.append(
            target_far
        )

    if len(
        set(normalized_fars)
    ) != len(
        normalized_fars
    ):
        raise ValueError(
            "FAR targets must be unique."
        )

    return normalized_fars


def _validate_verification_curve_inputs(
    scores,
    labels_pair,
):
    """
    Validate verification scores and binary pair labels.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )
    labels_pair = np.asarray(
        labels_pair
    )

    if scores.ndim != 1:
        raise ValueError(
            "Verification scores must be "
            "one-dimensional."
        )

    if labels_pair.ndim != 1:
        raise ValueError(
            "Verification pair labels must be "
            "one-dimensional."
        )

    if len(scores) != len(
        labels_pair
    ):
        raise ValueError(
            "Verification scores and labels "
            "must have equal lengths."
        )

    if len(scores) == 0:
        raise ValueError(
            "Verification scores cannot be empty."
        )

    if not np.all(
        np.isfinite(
            scores
        )
    ):
        raise ValueError(
            "Verification scores must be finite."
        )

    unique_labels = set(
        np.unique(
            labels_pair
        ).tolist()
    )

    if not unique_labels.issubset(
        {
            0,
            1,
            False,
            True,
        }
    ):
        raise ValueError(
            "Verification pair labels must "
            "contain only binary values 0 and 1."
        )

    if unique_labels != {
        0,
        1,
    }:
        raise ValueError(
            "Verification curves require both "
            "genuine and impostor comparisons."
        )

    return (
        scores,
        labels_pair.astype(
            int,
            copy=False,
        ),
    )



def _interpolate_equal_error_rate(
    false_accept_rates,
    false_reject_rates,
):
    """
    Linearly interpolate the empirical FAR/FRR crossing.
    """
    false_accept_rates = np.asarray(
        false_accept_rates,
        dtype=float,
    )
    false_reject_rates = np.asarray(
        false_reject_rates,
        dtype=float,
    )

    if (
        false_accept_rates.ndim != 1
        or false_reject_rates.ndim != 1
        or false_accept_rates.shape
        != false_reject_rates.shape
        or len(false_accept_rates) < 2
    ):
        raise ValueError(
            "EER requires aligned one-dimensional FAR and FRR "
            "arrays with at least two operating points."
        )

    if not (
        np.all(
            np.isfinite(
                false_accept_rates
            )
        )
        and np.all(
            np.isfinite(
                false_reject_rates
            )
        )
    ):
        raise ValueError(
            "EER requires finite FAR and FRR values."
        )

    difference = (
        false_accept_rates
        - false_reject_rates
    )

    exact = np.flatnonzero(
        difference == 0.0
    )

    if len(exact) > 0:
        eer = float(
            false_accept_rates[
                exact[0]
            ]
        )

    else:
        crossing_intervals = (
            np.flatnonzero(
                difference[:-1]
                * difference[1:]
                < 0.0
            )
        )

        if len(crossing_intervals) == 0:
            raise ValueError(
                "The empirical FAR and FRR curves do not "
                "contain an EER crossing."
            )

        lower = int(
            crossing_intervals[0]
        )
        upper = lower + 1

        lower_difference = float(
            difference[lower]
        )
        upper_difference = float(
            difference[upper]
        )

        denominator = (
            upper_difference
            - lower_difference
        )

        if denominator == 0.0:
            raise ValueError(
                "The EER crossing interval is degenerate."
            )

        fraction = (
            -lower_difference
            / denominator
        )

        interpolated_far = (
            false_accept_rates[lower]
            + fraction
            * (
                false_accept_rates[upper]
                - false_accept_rates[lower]
            )
        )

        interpolated_frr = (
            false_reject_rates[lower]
            + fraction
            * (
                false_reject_rates[upper]
                - false_reject_rates[lower]
            )
        )

        eer = float(
            0.5
            * (
                interpolated_far
                + interpolated_frr
            )
        )

    if (
        not np.isfinite(eer)
        or not 0.0 <= eer <= 1.0
    ):
        raise ValueError(
            "Interpolated EER is outside the valid "
            "probability range."
        )

    return eer


def _build_verification_curve_artifacts(
    scores,
    labels_pair,
    target_fars=None,
):
    """
    Build ROC, DET, comparison-count, and multi-FAR artifacts.

    Each reported operating point is selected from an observed ROC
    threshold with FAR no greater than the requested target. This avoids
    presenting an interpolated TAR as though it corresponded to a directly
    observed decision threshold.
    """
    (
        scores,
        labels_pair,
    ) = _validate_verification_curve_inputs(
        scores,
        labels_pair,
    )

    target_fars = (
        _normalize_target_fars(
            target_fars
        )
    )

    (
        false_accept_rates,
        true_accept_rates,
        roc_thresholds,
    ) = roc_curve(
        labels_pair,
        scores,
        drop_intermediate=False,
    )

    (
        det_false_accept_rates,
        det_false_reject_rates,
        det_thresholds,
    ) = det_curve(
        labels_pair,
        scores,
    )

    false_reject_rates = (
        1.0
        - true_accept_rates
    )

    genuine_count = int(
        np.sum(
            labels_pair == 1
        )
    )
    impostor_count = int(
        np.sum(
            labels_pair == 0
        )
    )

    minimum_nonzero_far = (
        1.0
        / impostor_count
    )

    eer = (
        _interpolate_equal_error_rate(
            false_accept_rates,
            false_reject_rates,
        )
    )

    tolerance = (
        np.finfo(float).eps
        * 32.0
    )

    operating_points = []

    for target_far in target_fars:
        far_percentage = (
            target_far
            * 100.0
        )

        if minimum_nonzero_far > target_far:
            operating_points.append(
                {
                    "name": (
                        "TAR@"
                        f"{far_percentage:g}%FAR"
                    ),
                    "target_far": float(
                        target_far
                    ),
                    "observed_far": None,
                    "tar": None,
                    "frr": None,
                    "threshold": None,
                    "empirically_resolvable": False,
                }
            )

            continue

        eligible_indices = np.flatnonzero(
            false_accept_rates
            <= (
                target_far
                + tolerance
            )
        )

        if len(eligible_indices) == 0:
            raise RuntimeError(
                "ROC construction did not produce "
                "an operating point at or below "
                "the requested FAR."
            )

        eligible_tars = (
            true_accept_rates[
                eligible_indices
            ]
        )

        selected_index = int(
            eligible_indices[
                np.argmax(
                    eligible_tars
                )
            ]
        )

        operating_points.append(
            {
                "name": (
                    "TAR@"
                    f"{far_percentage:g}%FAR"
                ),
                "target_far": float(
                    target_far
                ),
                "observed_far": float(
                    false_accept_rates[
                        selected_index
                    ]
                ),
                "tar": float(
                    true_accept_rates[
                        selected_index
                    ]
                ),
                "frr": float(
                    false_reject_rates[
                        selected_index
                    ]
                ),
                "threshold": (
                    _finite_float_or_none(
                        roc_thresholds[
                            selected_index
                        ]
                    )
                ),
                "empirically_resolvable": bool(
                    minimum_nonzero_far
                    <= target_far
                ),
            }
        )

    return {
        "type": "verification",
        "score_direction": (
            "higher_is_more_genuine"
        ),
        "comparison_counts": {
            "total": int(
                len(labels_pair)
            ),
            "genuine": genuine_count,
            "impostor": impostor_count,
            "minimum_nonzero_far": float(
                minimum_nonzero_far
            ),
        },
        "eer": float(
            eer
        ),
        "roc_auc": float(
            auc(
                false_accept_rates,
                true_accept_rates,
            )
        ),
        "operating_points": (
            operating_points
        ),
        "roc_curve": {
            "false_accept_rates": [
                float(value)
                for value in (
                    false_accept_rates
                )
            ],
            "true_accept_rates": [
                float(value)
                for value in (
                    true_accept_rates
                )
            ],
            "false_reject_rates": [
                float(value)
                for value in (
                    false_reject_rates
                )
            ],
            "thresholds": [
                _finite_float_or_none(
                    value
                )
                for value in (
                    roc_thresholds
                )
            ],
        },
        "det_curve": {
            "false_accept_rates": [
                float(value)
                for value in (
                    det_false_accept_rates
                )
            ],
            "false_reject_rates": [
                float(value)
                for value in (
                    det_false_reject_rates
                )
            ],
            "thresholds": [
                _finite_float_or_none(
                    value
                )
                for value in (
                    det_thresholds
                )
            ],
        },
    }


def _build_identification_curve_artifacts(
    score_matrix,
    true_labels,
):
    """
    Build the complete CMC curve for closed-set identification.

    ``true_labels`` must contain gallery-column indices, matching the
    convention already used by ``_compute_metrics_identification``.
    """
    score_matrix = np.asarray(
        score_matrix,
        dtype=float,
    )
    true_labels = np.asarray(
        true_labels
    )

    if score_matrix.ndim != 2:
        raise ValueError(
            "Identification scores must be "
            "a two-dimensional matrix."
        )

    if true_labels.ndim != 1:
        raise ValueError(
            "Identification labels must be "
            "one-dimensional."
        )

    if len(score_matrix) != len(
        true_labels
    ):
        raise ValueError(
            "Identification scores and labels "
            "must contain the same number "
            "of probes."
        )

    if score_matrix.shape[0] == 0:
        raise ValueError(
            "Identification scores cannot "
            "contain zero probes."
        )

    if score_matrix.shape[1] == 0:
        raise ValueError(
            "Identification scores cannot "
            "contain an empty gallery."
        )

    if not np.all(
        np.isfinite(
            score_matrix
        )
    ):
        raise ValueError(
            "Identification scores must be finite."
        )

    try:
        numeric_labels = true_labels.astype(
            float
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Identification labels must "
            "be integer gallery indices."
        ) from error

    if not np.all(
        np.isfinite(
            numeric_labels
        )
    ):
        raise ValueError(
            "Identification labels must be finite."
        )

    if not np.all(
        numeric_labels
        == np.floor(
            numeric_labels
        )
    ):
        raise ValueError(
            "Identification labels must "
            "be integer gallery indices."
        )

    true_labels = numeric_labels.astype(
        int
    )

    gallery_size = int(
        score_matrix.shape[1]
    )

    if np.any(
        true_labels < 0
    ) or np.any(
        true_labels
        >= gallery_size
    ):
        raise ValueError(
            "Identification labels must refer "
            "to valid gallery columns."
        )

    sorted_gallery_indices = np.argsort(
        score_matrix,
        axis=1,
    )[:, ::-1]

    correct_match_mask = (
        sorted_gallery_indices
        == true_labels[:, None]
    )

    correct_match_ranks = (
        np.argmax(
            correct_match_mask,
            axis=1,
        )
        + 1
    )

    ranks = np.arange(
        1,
        gallery_size + 1,
        dtype=int,
    )

    identification_rates = np.asarray(
        [
            np.mean(
                correct_match_ranks
                <= rank
            )
            for rank in ranks
        ],
        dtype=float,
    )

    effective_rank_5 = min(
        5,
        gallery_size,
    )

    return {
        "type": "identification",
        "probe_count": int(
            len(true_labels)
        ),
        "gallery_size": gallery_size,
        "maximum_meaningful_rank": (
            gallery_size
        ),
        "rank_1_accuracy": float(
            identification_rates[0]
        ),
        "rank_5_accuracy": float(
            identification_rates[
                effective_rank_5 - 1
            ]
        ),
        "effective_rank_5": int(
            effective_rank_5
        ),
        "correct_match_ranks": [
            int(value)
            for value in (
                correct_match_ranks
            )
        ],
        "cmc_curve": {
            "ranks": [
                int(value)
                for value in ranks
            ],
            "identification_rates": [
                float(value)
                for value in (
                    identification_rates
                )
            ],
        },
    }



def _compute_metrics_verification(
    scores,
    labels_pair,
):
    """
    Calculate verification EER, AUC, d-prime, and TAR at 0.1% FAR.

    EER is linearly interpolated across the empirical FAR/FRR crossing.
    TAR uses an observed threshold with FAR no greater than 0.1% and is
    unavailable when the impostor count cannot resolve that FAR.
    """
    (
        scores,
        labels_pair,
    ) = _validate_verification_curve_inputs(
        scores,
        labels_pair,
    )

    artifacts = (
        _build_verification_curve_artifacts(
            scores,
            labels_pair,
            target_fars=[
                0.001,
            ],
        )
    )

    eer = float(
        artifacts["eer"]
    )

    roc_auc = float(
        artifacts["roc_auc"]
    )

    tar_at_far = (
        artifacts[
            "operating_points"
        ][0]["tar"]
    )

    genuine_scores = scores[
        labels_pair == 1
    ]

    impostor_scores = scores[
        labels_pair == 0
    ]

    mu_genuine = np.mean(
        genuine_scores
    )
    sigma_genuine = np.std(
        genuine_scores
    )

    mu_impostor = np.mean(
        impostor_scores
    )
    sigma_impostor = np.std(
        impostor_scores
    )

    d_prime = float(
        abs(
            mu_genuine
            - mu_impostor
        )
        / np.sqrt(
            0.5
            * (
                sigma_genuine ** 2
                + sigma_impostor ** 2
            )
            + 1e-10
        )
    )

    tar_text = (
        "N/A"
        if tar_at_far is None
        else f"{tar_at_far:.4f}"
    )

    print(
        "[RESULT] "
        f"EER: {eer:.4f} | "
        f"AUC: {roc_auc:.4f} | "
        f"d': {d_prime:.4f} | "
        f"TAR@0.1%FAR: {tar_text}"
    )

    return (
        eer,
        roc_auc,
        d_prime,
        tar_at_far,
    )


def _compute_metrics_identification(
    preds_probs,
    true_labels,
):
    """
    Calculate Rank-1 and Rank-5 identification accuracy from the CMC curve.

    The public return signature remains unchanged.
    """
    artifacts = (
        _build_identification_curve_artifacts(
            preds_probs,
            true_labels,
        )
    )

    rank1 = artifacts[
        "rank_1_accuracy"
    ]
    rank5 = artifacts[
        "rank_5_accuracy"
    ]

    print(
        "[RESULT] "
        f"Rank-1 Acc: {rank1:.4f} | "
        f"Rank-5 Acc: {rank5:.4f}"
    )

    return (
        rank1,
        rank5,
    )



def _group_verification_indices_by_label(labels):
    """
    Group verification sample indices by identity label.

    Indices retain their original observation order.
    """
    labels = np.asarray(labels)

    if labels.ndim != 1:
        raise ValueError(
            "Verification identity labels must be one-dimensional."
        )

    groups = collections.defaultdict(list)

    for index, label in enumerate(labels):
        groups[label].append(index)

    return {
        label: np.asarray(
            indices,
            dtype=np.int64,
        )
        for label, indices in groups.items()
    }


def _sample_unique_integer_ranks(
    population_size,
    sample_size,
    rng,
):
    """
    Uniformly sample integer ranks without replacement.

    Floyd's algorithm uses memory proportional to the requested sample,
    rather than to the full candidate population.
    """
    if (
        isinstance(population_size, (bool, np.bool_))
        or not isinstance(
            population_size,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "population_size must be an integer."
        )

    if (
        isinstance(sample_size, (bool, np.bool_))
        or not isinstance(
            sample_size,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "sample_size must be an integer."
        )

    population_size = int(population_size)
    sample_size = int(sample_size)

    if population_size < 0:
        raise ValueError(
            "population_size must be non-negative."
        )

    if not 0 <= sample_size <= population_size:
        raise ValueError(
            "sample_size must satisfy "
            "0 <= sample_size <= population_size."
        )

    if sample_size == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if sample_size == population_size:
        return np.arange(
            population_size,
            dtype=np.int64,
        )

    selected = set()

    for upper_bound in range(
        population_size - sample_size,
        population_size,
    ):
        draw = int(
            rng.integers(
                0,
                upper_bound + 1,
            )
        )

        if draw in selected:
            selected.add(
                upper_bound
            )
        else:
            selected.add(
                draw
            )

    if len(selected) != sample_size:
        raise RuntimeError(
            "Unique-rank sampling produced an "
            "unexpected sample count."
        )

    return np.sort(
        np.fromiter(
            selected,
            dtype=np.int64,
            count=sample_size,
        )
    )


def _map_offsets_excluding_indices(
    offsets,
    forbidden_indices,
    lower_bound,
    upper_bound,
):
    """
    Map offsets in a filtered integer range back to original indices.

    The valid population is [lower_bound, upper_bound) with every index
    listed in forbidden_indices removed.
    """
    offsets = np.asarray(
        offsets,
        dtype=np.int64,
    )

    forbidden_indices = np.asarray(
        forbidden_indices,
        dtype=np.int64,
    )

    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)

    if offsets.ndim != 1:
        raise ValueError(
            "offsets must be one-dimensional."
        )

    if forbidden_indices.ndim != 1:
        raise ValueError(
            "forbidden_indices must be one-dimensional."
        )

    if lower_bound < 0:
        raise ValueError(
            "lower_bound must be non-negative."
        )

    if upper_bound < lower_bound:
        raise ValueError(
            "upper_bound must not be smaller than lower_bound."
        )

    if offsets.size == 0:
        return np.empty(
            0,
            dtype=np.int64,
        )

    if np.any(offsets < 0):
        raise ValueError(
            "offsets must be non-negative."
        )

    if forbidden_indices.size:
        if np.any(
            forbidden_indices[:-1]
            > forbidden_indices[1:]
        ):
            raise ValueError(
                "forbidden_indices must be sorted."
            )

        forbidden_start = int(
            np.searchsorted(
                forbidden_indices,
                lower_bound,
                side="left",
            )
        )

        forbidden_end = int(
            np.searchsorted(
                forbidden_indices,
                upper_bound,
                side="left",
            )
        )

        forbidden_in_range = (
            forbidden_indices[
                forbidden_start:
                forbidden_end
            ]
        )
    else:
        forbidden_in_range = (
            forbidden_indices
        )

    available_count = (
        upper_bound
        - lower_bound
        - len(forbidden_in_range)
    )

    if np.any(
        offsets >= available_count
    ):
        raise ValueError(
            "An offset exceeds the available filtered range."
        )

    # The kth allowed index can never be before lower_bound + k.
    low = (
        lower_bound
        + offsets
    ).astype(
        np.int64,
        copy=True,
    )

    high = np.full(
        offsets.shape,
        upper_bound - 1,
        dtype=np.int64,
    )

    # Find the smallest original index for which the number of allowed
    # positions encountered is greater than the requested zero-based offset.
    while np.any(low < high):
        middle = (
            low
            + (high - low) // 2
        )

        excluded_through_middle = (
            np.searchsorted(
                forbidden_in_range,
                middle,
                side="right",
            )
        )

        allowed_through_middle = (
            middle
            - lower_bound
            + 1
            - excluded_through_middle
        )

        move_left = (
            allowed_through_middle
            > offsets
        )

        high = np.where(
            move_left,
            middle,
            high,
        )

        low = np.where(
            move_left,
            low,
            middle + 1,
        )

    result = low

    if forbidden_in_range.size:
        if np.any(
            np.isin(
                result,
                forbidden_in_range,
            )
        ):
            raise RuntimeError(
                "Filtered-offset mapping returned "
                "a forbidden index."
            )

    return result


def _sample_impostor_pair_indices(
    labels1,
    labels2=None,
    max_impostor_pairs=1000000,
    pair_sampling_seed=42,
):
    """
    Uniformly sample impostor comparisons without replacement.

    One-set evaluation uses only unordered pairs i < j. Two-set evaluation
    treats every row-by-column Cartesian cell as a distinct comparison.

    The full comparison matrix is never materialized.
    """
    labels1 = np.asarray(labels1)

    if labels1.ndim != 1:
        raise ValueError(
            "labels1 must be one-dimensional."
        )

    match_two_sets = (
        labels2 is not None
    )

    if match_two_sets:
        labels2 = np.asarray(labels2)

        if labels2.ndim != 1:
            raise ValueError(
                "labels2 must be one-dimensional."
            )
    else:
        labels2 = labels1

    if (
        isinstance(
            max_impostor_pairs,
            (bool, np.bool_),
        )
        or not isinstance(
            max_impostor_pairs,
            (int, np.integer),
        )
        or int(max_impostor_pairs) < 1
    ):
        raise ValueError(
            "max_impostor_pairs must be a positive integer."
        )

    if (
        isinstance(
            pair_sampling_seed,
            (bool, np.bool_),
        )
        or not isinstance(
            pair_sampling_seed,
            (int, np.integer),
        )
        or int(pair_sampling_seed) < 0
    ):
        raise ValueError(
            "pair_sampling_seed must be a non-negative integer."
        )

    row_count = len(labels1)
    column_count = len(labels2)

    if (
        row_count == 0
        or column_count == 0
    ):
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            0,
        )

    column_groups = (
        _group_verification_indices_by_label(
            labels2
        )
    )

    row_impostor_counts = np.zeros(
        row_count,
        dtype=np.int64,
    )

    if match_two_sets:
        for row_index, label in enumerate(
            labels1
        ):
            same_identity_columns = (
                column_groups.get(
                    label
                )
            )

            same_identity_count = (
                0
                if same_identity_columns is None
                else len(same_identity_columns)
            )

            row_impostor_counts[
                row_index
            ] = (
                column_count
                - same_identity_count
            )

    else:
        identity_counts = {
            label: len(indices)
            for label, indices
            in column_groups.items()
        }

        seen_counts = (
            collections.defaultdict(int)
        )

        for row_index, label in enumerate(
            labels1
        ):
            later_samples = (
                row_count
                - row_index
                - 1
            )

            later_same_identity = (
                identity_counts[label]
                - seen_counts[label]
                - 1
            )

            row_impostor_counts[
                row_index
            ] = (
                later_samples
                - later_same_identity
            )

            seen_counts[label] += 1

    if np.any(
        row_impostor_counts < 0
    ):
        raise RuntimeError(
            "Impostor row counts cannot be negative."
        )

    row_ends = np.cumsum(
        row_impostor_counts,
        dtype=np.int64,
    )

    total_impostor_pairs = (
        int(row_ends[-1])
        if row_ends.size
        else 0
    )

    sample_size = min(
        total_impostor_pairs,
        int(max_impostor_pairs),
    )

    if sample_size == 0:
        return (
            np.empty(
                0,
                dtype=np.int64,
            ),
            np.empty(
                0,
                dtype=np.int64,
            ),
            total_impostor_pairs,
        )

    rng = np.random.default_rng(
        int(pair_sampling_seed)
    )

    sampled_ranks = (
        _sample_unique_integer_ranks(
            total_impostor_pairs,
            sample_size,
            rng,
        )
    )

    row_indices = np.searchsorted(
        row_ends,
        sampled_ranks,
        side="right",
    ).astype(
        np.int64,
        copy=False,
    )

    row_starts = np.zeros_like(
        row_ends
    )

    if len(row_starts) > 1:
        row_starts[1:] = (
            row_ends[:-1]
        )

    offsets_within_row = (
        sampled_ranks
        - row_starts[
            row_indices
        ]
    )

    column_indices = np.empty(
        sample_size,
        dtype=np.int64,
    )

    unique_rows, group_starts = (
        np.unique(
            row_indices,
            return_index=True,
        )
    )

    group_ends = np.concatenate(
        (
            group_starts[1:],
            np.asarray(
                [sample_size],
                dtype=np.int64,
            ),
        )
    )

    empty_forbidden = np.empty(
        0,
        dtype=np.int64,
    )

    for row_index, start, end in zip(
        unique_rows,
        group_starts,
        group_ends,
    ):
        row_index = int(row_index)
        start = int(start)
        end = int(end)

        label = labels1[
            row_index
        ]

        forbidden_indices = (
            column_groups.get(
                label,
                empty_forbidden,
            )
        )

        lower_bound = (
            0
            if match_two_sets
            else row_index + 1
        )

        column_indices[
            start:end
        ] = (
            _map_offsets_excluding_indices(
                offsets_within_row[
                    start:end
                ],
                forbidden_indices,
                lower_bound=lower_bound,
                upper_bound=column_count,
            )
        )

    if not np.all(
        labels1[row_indices]
        != labels2[column_indices]
    ):
        raise RuntimeError(
            "Impostor sampling generated a "
            "same-identity comparison."
        )

    if (
        not match_two_sets
        and not np.all(
            row_indices
            < column_indices
        )
    ):
        raise RuntimeError(
            "One-set verification must use only i < j pairs."
        )

    return (
        row_indices,
        column_indices,
        total_impostor_pairs,
    )



def _resolve_pair_sampling_arguments(
    pair_sampling_mode=None,
    pair_sampling_budget=None,
    sampling_mode=None,
    num_pairs=None,
):
    """
    Resolve canonical verification pair-sampling arguments and legacy aliases.

    ``pair_sampling_mode`` and ``pair_sampling_budget`` are the canonical
    names. ``sampling_mode`` and ``num_pairs`` remain supported as aliases.
    """
    if pair_sampling_mode is None:
        resolved_mode = (
            "all"
            if sampling_mode is None
            else sampling_mode
        )
    else:
        resolved_mode = pair_sampling_mode

        if (
            sampling_mode is not None
            and sampling_mode
            != pair_sampling_mode
        ):
            raise ValueError(
                "Conflicting pair sampling modes were provided "
                "through pair_sampling_mode and sampling_mode."
            )

    valid_modes = {
        "all",
        "all_genuine",
        "balanced",
        "random",
    }

    if resolved_mode not in valid_modes:
        raise ValueError(
            "Unknown verification pair sampling mode: "
            f"{resolved_mode!r}."
        )

    if pair_sampling_budget is None:
        resolved_budget = (
            10000
            if num_pairs is None
            else num_pairs
        )
    else:
        resolved_budget = pair_sampling_budget

        if (
            num_pairs is not None
            and num_pairs
            != pair_sampling_budget
        ):
            raise ValueError(
                "Conflicting pair sampling budgets were provided "
                "through pair_sampling_budget and num_pairs."
            )

    if (
        isinstance(
            resolved_budget,
            (bool, np.bool_),
        )
        or not isinstance(
            resolved_budget,
            (int, np.integer),
        )
        or int(resolved_budget) < 1
    ):
        raise ValueError(
            "pair_sampling_budget must be a positive integer."
        )

    return (
        resolved_mode,
        int(resolved_budget),
    )


def _verification_pair_rng(
    pair_sampling_seed,
):
    """
    Return the RNG used for stochastic verification-pair sampling.

    A supplied seed creates an isolated Generator. ``None`` preserves the
    legacy global NumPy RNG behavior required by training-validation callers.
    """
    if pair_sampling_seed is None:
        return np.random

    if (
        isinstance(
            pair_sampling_seed,
            (bool, np.bool_),
        )
        or not isinstance(
            pair_sampling_seed,
            (int, np.integer),
        )
        or int(pair_sampling_seed) < 0
    ):
        raise ValueError(
            "pair_sampling_seed must be a non-negative integer."
        )

    return np.random.default_rng(
        int(pair_sampling_seed)
    )


def _compute_aligned_pair_scores(
    embeddings1,
    embeddings2,
    method="cosine",
):
    """
    Score corresponding rows from two equally shaped embedding arrays.
    """
    embeddings1 = np.asarray(
        embeddings1
    )

    embeddings2 = np.asarray(
        embeddings2
    )

    if (
        embeddings1.ndim != 2
        or embeddings2.ndim != 2
        or embeddings1.shape
        != embeddings2.shape
    ):
        raise ValueError(
            "Aligned pair scoring requires equally shaped "
            "two-dimensional embedding arrays."
        )

    if method == "cosine":
        numerator = np.sum(
            embeddings1
            * embeddings2,
            axis=1,
        )

        denominator = (
            (
                np.linalg.norm(
                    embeddings1,
                    axis=1,
                )
                + 1e-10
            )
            * (
                np.linalg.norm(
                    embeddings2,
                    axis=1,
                )
                + 1e-10
            )
        )

        return (
            numerator
            / denominator
        )

    if method == "euclidean":
        return -np.linalg.norm(
            embeddings1
            - embeddings2,
            axis=1,
        )

    if method == "manhattan":
        return -np.sum(
            np.abs(
                embeddings1
                - embeddings2
            ),
            axis=1,
        )

    if method == "correlation":
        centered1 = (
            embeddings1
            - np.mean(
                embeddings1,
                axis=1,
                keepdims=True,
            )
        )

        centered2 = (
            embeddings2
            - np.mean(
                embeddings2,
                axis=1,
                keepdims=True,
            )
        )

        numerator = np.sum(
            centered1
            * centered2,
            axis=1,
        )

        denominator = (
            np.linalg.norm(
                centered1,
                axis=1,
            )
            * np.linalg.norm(
                centered2,
                axis=1,
            )
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            return (
                numerator
                / denominator
            )

    raise ValueError(
        f"Unknown matching method: {method}"
    )


def _score_selected_pair_indices(
    embeddings1,
    embeddings2,
    row_indices,
    column_indices,
    matching_method,
    chunk_size=65536,
):
    """
    Score selected comparisons without constructing a Cartesian score matrix.
    """
    embeddings1 = np.asarray(
        embeddings1
    )

    embeddings2 = np.asarray(
        embeddings2
    )

    row_indices = np.asarray(
        row_indices,
        dtype=np.int64,
    )

    column_indices = np.asarray(
        column_indices,
        dtype=np.int64,
    )

    if row_indices.shape != column_indices.shape:
        raise ValueError(
            "Selected pair-index arrays must have equal shape."
        )

    scores = np.empty(
        len(row_indices),
        dtype=float,
    )

    for start in range(
        0,
        len(row_indices),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(row_indices),
        )

        scores[
            start:end
        ] = _compute_aligned_pair_scores(
            embeddings1[
                row_indices[start:end]
            ],
            embeddings2[
                column_indices[start:end]
            ],
            method=matching_method,
        )

    return scores


def _score_all_genuine_comparisons(
    embeddings1,
    labels1,
    embeddings2,
    labels2,
    match_two_sets,
    matching_method,
    chunk_size=65536,
):
    """
    Score every genuine verification comparison exactly once.

    One-set evaluation scores unordered same-identity pairs. Two-set
    evaluation scores every same-identity Cartesian cell. Only bounded
    chunks of embeddings are materialized while scoring.
    """
    embeddings1 = np.asarray(
        embeddings1
    )

    embeddings2 = np.asarray(
        embeddings2
    )

    labels1 = np.asarray(
        labels1
    )

    labels2 = np.asarray(
        labels2
    )

    if (
        embeddings1.ndim != 2
        or embeddings2.ndim != 2
    ):
        raise ValueError(
            "Verification embeddings must be two-dimensional."
        )

    if (
        embeddings1.shape[1]
        != embeddings2.shape[1]
    ):
        raise ValueError(
            "Verification embedding sets must have "
            "the same feature dimension."
        )

    groups1 = (
        _group_verification_indices_by_label(
            labels1
        )
    )

    groups2 = (
        _group_verification_indices_by_label(
            labels2
        )
    )

    if match_two_sets:
        genuine_count = sum(
            len(indices1)
            * len(
                groups2.get(
                    label,
                    (),
                )
            )
            for label, indices1
            in groups1.items()
        )
    else:
        genuine_count = sum(
            len(indices)
            * (
                len(indices) - 1
            )
            // 2
            for indices
            in groups1.values()
        )

    scores = np.empty(
        genuine_count,
        dtype=float,
    )

    cursor = 0

    if match_two_sets:
        for label, row_indices in (
            groups1.items()
        ):
            column_indices = (
                groups2.get(
                    label
                )
            )

            if column_indices is None:
                continue

            for row_index in row_indices:
                for start in range(
                    0,
                    len(column_indices),
                    chunk_size,
                ):
                    selected_columns = (
                        column_indices[
                            start:
                            start + chunk_size
                        ]
                    )

                    count = len(
                        selected_columns
                    )

                    left_embeddings = (
                        np.broadcast_to(
                            embeddings1[
                                int(row_index)
                            ],
                            (
                                count,
                                embeddings1.shape[1],
                            ),
                        )
                    )

                    scores[
                        cursor:
                        cursor + count
                    ] = (
                        _compute_aligned_pair_scores(
                            left_embeddings,
                            embeddings2[
                                selected_columns
                            ],
                            method=matching_method,
                        )
                    )

                    cursor += count

    else:
        for identity_indices in (
            groups1.values()
        ):
            for position, row_index in enumerate(
                identity_indices[:-1]
            ):
                later_indices = (
                    identity_indices[
                        position + 1:
                    ]
                )

                for start in range(
                    0,
                    len(later_indices),
                    chunk_size,
                ):
                    selected_columns = (
                        later_indices[
                            start:
                            start + chunk_size
                        ]
                    )

                    count = len(
                        selected_columns
                    )

                    left_embeddings = (
                        np.broadcast_to(
                            embeddings1[
                                int(row_index)
                            ],
                            (
                                count,
                                embeddings1.shape[1],
                            ),
                        )
                    )

                    scores[
                        cursor:
                        cursor + count
                    ] = (
                        _compute_aligned_pair_scores(
                            left_embeddings,
                            embeddings2[
                                selected_columns
                            ],
                            method=matching_method,
                        )
                    )

                    cursor += count

    if cursor != genuine_count:
        raise RuntimeError(
            "Genuine comparison scoring produced "
            "an unexpected comparison count."
        )

    return scores



def _generate_pairs(
    embeddings1,
    labels1,
    embeddings2=None,
    labels2=None,
    num_pairs=None,
    sampling_mode=None,
    matching_method="cosine",
    *,
    pair_sampling_budget=None,
    pair_sampling_mode=None,
    max_impostor_pairs=1000000,
    pair_sampling_seed=None,
):
    """
    Generate verification scores and binary comparison labels.

    Canonical pair-sampling arguments are ``pair_sampling_mode`` and
    ``pair_sampling_budget``. ``sampling_mode`` and ``num_pairs`` remain
    supported as legacy aliases.

    Modes:
      - all: evaluate the complete comparison space.
      - all_genuine: evaluate every genuine comparison and uniformly sample
        impostor comparisons without replacement up to max_impostor_pairs.
      - balanced: sample equal genuine and impostor classes using the pair
        sampling budget.
      - random: draw comparison attempts at random using the pair sampling
        budget.

    In one-set evaluation, comparisons are unordered and self-comparisons are
    excluded. In two-set evaluation, each probe-by-enrollment Cartesian cell
    is a distinct comparison.
    """
    (
        resolved_mode,
        resolved_budget,
    ) = _resolve_pair_sampling_arguments(
        pair_sampling_mode=(
            pair_sampling_mode
        ),
        pair_sampling_budget=(
            pair_sampling_budget
        ),
        sampling_mode=sampling_mode,
        num_pairs=num_pairs,
    )

    embeddings1 = np.asarray(
        embeddings1
    )

    labels1 = np.asarray(
        labels1
    )

    match_two_sets = (
        embeddings2 is not None
    )

    if match_two_sets:
        embeddings2 = np.asarray(
            embeddings2
        )

        if labels2 is None:
            raise ValueError(
                "labels2 is required when embeddings2 is provided."
            )

        labels2 = np.asarray(
            labels2
        )
    else:
        embeddings2 = embeddings1
        labels2 = labels1

    if len(embeddings1) != len(labels1):
        raise ValueError(
            "embeddings1 and labels1 must have equal length."
        )

    if len(embeddings2) != len(labels2):
        raise ValueError(
            "embeddings2 and labels2 must have equal length."
        )

    if (
        len(labels1) == 0
        or len(labels2) == 0
    ):
        return (
            np.array([]),
            np.array([]),
        )

    if resolved_mode == "all_genuine":
        resolved_pair_seed = (
            42
            if pair_sampling_seed is None
            else pair_sampling_seed
        )

        genuine_scores = (
            _score_all_genuine_comparisons(
                embeddings1,
                labels1,
                embeddings2,
                labels2,
                match_two_sets=(
                    match_two_sets
                ),
                matching_method=(
                    matching_method
                ),
            )
        )

        (
            impostor_rows,
            impostor_columns,
            total_impostor_pairs,
        ) = _sample_impostor_pair_indices(
            labels1,
            (
                labels2
                if match_two_sets
                else None
            ),
            max_impostor_pairs=(
                max_impostor_pairs
            ),
            pair_sampling_seed=(
                resolved_pair_seed
            ),
        )

        impostor_scores = (
            _score_selected_pair_indices(
                embeddings1,
                embeddings2,
                impostor_rows,
                impostor_columns,
                matching_method=(
                    matching_method
                ),
            )
        )

        scores = np.concatenate(
            (
                genuine_scores,
                impostor_scores,
            )
        )

        labels_pair = np.concatenate(
            (
                np.ones(
                    len(genuine_scores),
                    dtype=int,
                ),
                np.zeros(
                    len(impostor_scores),
                    dtype=int,
                ),
            )
        )

        print(
            "[INFO] Verification pair generation: "
            "all genuine comparisons retained; "
            f"{len(impostor_scores):,} of "
            f"{total_impostor_pairs:,} available "
            "impostor comparisons evaluated."
        )

        return (
            scores,
            labels_pair,
        )

    scores = []
    labels_pair = []

    if resolved_mode == "all":
        print(
            "[INFO] generating ALL pairs "
            "(Full Matrix evaluation)..."
        )

        sim_matrix = _compute_score_matrix(
            embeddings1,
            embeddings2,
            method=matching_method,
        )

        truth_matrix = (
            labels1[:, None]
            == labels2[None, :]
        ).astype(int)

        if match_two_sets:
            scores = sim_matrix.flatten()
            labels_pair = (
                truth_matrix.flatten()
            )
        else:
            upper_tri = np.triu_indices(
                len(labels1),
                k=1,
            )

            scores = sim_matrix[
                upper_tri
            ]

            labels_pair = truth_matrix[
                upper_tri
            ]

        return (
            np.asarray(scores),
            np.asarray(labels_pair),
        )

    rng = _verification_pair_rng(
        pair_sampling_seed
    )

    if resolved_mode == "balanced":
        s1_idx = collections.defaultdict(
            list
        )

        s2_idx = collections.defaultdict(
            list
        )

        for index, label in enumerate(
            labels1
        ):
            s1_idx[label].append(
                index
            )

        for index, label in enumerate(
            labels2
        ):
            s2_idx[label].append(
                index
            )

        common_subjects = [
            subject
            for subject in s1_idx
            if subject in s2_idx
        ]

        if len(common_subjects) < 2:
            return (
                np.array([]),
                np.array([]),
            )

        if match_two_sets:
            genuine_subjects = (
                common_subjects
            )
        else:
            genuine_subjects = [
                subject
                for subject
                in common_subjects
                if len(
                    s1_idx[subject]
                ) >= 2
            ]

        if not genuine_subjects:
            return (
                np.array([]),
                np.array([]),
            )

        for _ in range(
            resolved_budget // 2
        ):
            subject = rng.choice(
                genuine_subjects
            )

            if match_two_sets:
                idx1 = int(
                    rng.choice(
                        s1_idx[subject]
                    )
                )

                idx2 = int(
                    rng.choice(
                        s2_idx[subject]
                    )
                )
            else:
                candidate_indices = (
                    np.asarray(
                        s1_idx[subject]
                    )
                )

                idx1, idx2 = (
                    rng.choice(
                        candidate_indices,
                        size=2,
                        replace=False,
                    )
                )

                idx1 = int(idx1)
                idx2 = int(idx2)

            score = _compute_pair_score(
                embeddings1[idx1],
                embeddings2[idx2],
                method=matching_method,
            )

            scores.append(
                score
            )

            labels_pair.append(
                1
            )

        all_s1 = list(
            s1_idx.keys()
        )

        all_s2 = list(
            s2_idx.keys()
        )

        for _ in range(
            resolved_budget // 2
        ):
            subject_a = rng.choice(
                all_s1
            )

            possible_b = [
                subject
                for subject in all_s2
                if subject != subject_a
            ]

            if not possible_b:
                continue

            subject_b = rng.choice(
                possible_b
            )

            idx1 = int(
                rng.choice(
                    s1_idx[subject_a]
                )
            )

            idx2 = int(
                rng.choice(
                    s2_idx[subject_b]
                )
            )

            score = _compute_pair_score(
                embeddings1[idx1],
                embeddings2[idx2],
                method=matching_method,
            )

            scores.append(
                score
            )

            labels_pair.append(
                0
            )

    elif resolved_mode == "random":
        indices1 = np.arange(
            len(labels1)
        )

        indices2 = np.arange(
            len(labels2)
        )

        for _ in range(
            resolved_budget
        ):
            idx1 = int(
                rng.choice(
                    indices1
                )
            )

            idx2 = int(
                rng.choice(
                    indices2
                )
            )

            if (
                not match_two_sets
                and idx1 == idx2
            ):
                continue

            score = _compute_pair_score(
                embeddings1[idx1],
                embeddings2[idx2],
                method=matching_method,
            )

            scores.append(
                score
            )

            labels_pair.append(
                int(
                    labels1[idx1]
                    == labels2[idx2]
                )
            )

    return (
        np.asarray(scores),
        np.asarray(labels_pair),
    )


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
    return_indices=False,
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

    Set ``return_indices=True`` to also receive the exact retained indices
    (relative to the supplied arrays) for aligning a parallel array.
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

    if return_indices:
        # ``final_indices`` are the exact samples retained, relative to the
        # arrays passed in, so a caller can align a parallel array such as
        # per-beat provenance with the filtered output.
        return x[final_indices], y[final_indices], final_indices

    return x[final_indices], y[final_indices]


PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DEFAULT_CACHE_DIR = os.path.join(
    "..",
    "ecg-biometrics-artifacts",
    "cache",
)

DEFAULT_RESULTS_DIR = os.path.join(
    "..",
    "ecg-biometrics-artifacts",
    "results",
)


def resolve_artifact_path(path):
    """
    Resolve an artifact path relative to the repository directory.

    Environment variables and user-home markers are expanded. Absolute
    paths are preserved.
    """
    if path is None:
        raise ValueError(
            "Artifact path cannot be None."
        )

    expanded_path = os.path.expandvars(
        os.path.expanduser(
            os.fspath(path)
        )
    )

    if not expanded_path.strip():
        raise ValueError(
            "Artifact path cannot be empty."
        )

    if not os.path.isabs(
        expanded_path
    ):
        expanded_path = os.path.join(
            PROJECT_ROOT,
            expanded_path,
        )

    return os.path.abspath(
        expanded_path
    )


_CACHE_RELEVANT_LOADER_ATTRIBUTES = (
    "data_split_mode",
    "num_beats",
    "num_beats_to_merge",
    "merge_strategy",
    "beat_merge_method",
    "signal_type",
    "train_sessions",
    "enroll_sessions",
    "enrol_sessions",
    "probe_sessions",
    "session_for_single_session_evaluation",
    "required_cross_sessions",
    "electrode_unit",
    "target_leads",
    "leads",
    "temporal_date_policy",
    "single_segment_range",
    "train_parts",
    "enrol_parts",
    "enroll_parts",
    "test_parts",
    "train_record_indices",
    "enroll_record_indices",
    "probe_record_indices",
    "only_healthy",
    "resolution",
    "limit_records",
    "sample_len",
    "data_root",
    "dataset_root",
)


# Loader options that are elided from the cache identity while they hold the
# neutral default listed here. This keeps cache entries valid across the
# introduction of a new option, while a deliberate non-default setting still
# forces regeneration.
_DEFAULT_ELIDED_LOADER_ATTRIBUTES = {
    "beat_merge_stride": 1,
    "temporal_guard_minutes": 0.0,
}


def _is_neutral_default(value, neutral_default):
    """
    Return True when a loader option still carries its neutral default.

    Booleans are compared by identity of type because ``True == 1`` would
    otherwise elide a genuinely different setting.
    """
    if isinstance(value, bool) != isinstance(neutral_default, bool):
        return False

    try:
        return bool(value == neutral_default)
    except (TypeError, ValueError):
        return False


def _build_loader_cache_identity(loader):
    """
    Build the effective dataset-loader identity used by data and weight caches.

    The identity records the configured dataset definition, merged
    preprocessing settings, and public loader attributes that can alter which
    records, channels, sessions, leads, or samples are returned.
    """
    if loader is None:
        return None

    loader_cfg = getattr(
        loader,
        "cfg",
        {},
    )

    dataset_config = {}
    effective_preprocessing = {}

    if isinstance(loader_cfg, dict):
        dataset_config = copy.deepcopy(
            loader_cfg
        )

        configured_preprocessing = dataset_config.pop(
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

    for attribute_name in _CACHE_RELEVANT_LOADER_ATTRIBUTES:
        if hasattr(loader, attribute_name):
            loader_settings[attribute_name] = copy.deepcopy(
                getattr(
                    loader,
                    attribute_name,
                )
            )

    for (
        attribute_name,
        neutral_default,
    ) in _DEFAULT_ELIDED_LOADER_ATTRIBUTES.items():
        if not hasattr(loader, attribute_name):
            continue

        attribute_value = getattr(
            loader,
            attribute_name,
        )

        if _is_neutral_default(
            attribute_value,
            neutral_default,
        ):
            continue

        loader_settings[attribute_name] = copy.deepcopy(
            attribute_value
        )

    return {
        "loader_class": type(loader).__name__,
        "root_dir": dataset_config.get(
            "root_dir"
        ),
        "dataset_config": dataset_config,
        "preprocessing": effective_preprocessing,
        "settings": loader_settings,
    }


def _fingerprint_array_collection(arrays):
    """
    Return a deterministic SHA-256 identity for named NumPy-compatible arrays.

    Shapes, dtypes, names, and complete array contents are included. Object
    arrays are serialized canonically rather than hashing process-specific
    object pointers.
    """
    digest = hashlib.sha256()
    array_metadata = {}

    for array_name in sorted(arrays):
        array = np.asarray(
            arrays[array_name]
        )

        metadata = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

        array_metadata[array_name] = metadata

        header = json.dumps(
            {
                "name": str(array_name),
                **metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        digest.update(
            len(header).to_bytes(
                8,
                byteorder="big",
            )
        )
        digest.update(header)

        if array.dtype.hasobject:
            payload = json.dumps(
                array.tolist(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")

            digest.update(
                len(payload).to_bytes(
                    8,
                    byteorder="big",
                )
            )
            digest.update(payload)

        else:
            contiguous_array = np.ascontiguousarray(
                array
            )
            byte_view = memoryview(
                contiguous_array
            ).cast("B")

            digest.update(
                len(byte_view).to_bytes(
                    8,
                    byteorder="big",
                )
            )

            chunk_size = 8 * 1024 * 1024

            for offset in range(
                0,
                len(byte_view),
                chunk_size,
            ):
                digest.update(
                    byte_view[
                        offset:
                        offset + chunk_size
                    ]
                )

    return {
        "sha256": digest.hexdigest(),
        "arrays": array_metadata,
    }


def _generate_config_hash(config_dict):
    """Creates a deterministic short hash from a dictionary of parameters."""
    # Convert dict to a sorted JSON string to ensure consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()[:12]

def _atomic_write_file(final_path, writer):
    """
    Write a file through a temporary path and atomically replace the target.

    The temporary file is created in the destination directory so
    ``os.replace`` remains atomic on the same filesystem.
    """
    final_path = os.fspath(final_path)
    directory = os.path.dirname(final_path) or "."

    os.makedirs(
        directory,
        exist_ok=True,
    )

    suffix = os.path.splitext(final_path)[1]

    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(final_path)}.",
        suffix=suffix,
    )

    os.close(file_descriptor)

    try:
        writer(temporary_path)
        os.replace(
            temporary_path,
            final_path,
        )

    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _remove_cache_files(*paths):
    """
    Remove unusable cache files without masking the original cache failure.
    """
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as error:
            print(
                "[WARN] Could not remove unusable cache file "
                f"'{path}': {error}"
            )

class CacheManager:
    def __init__(self, base_dir=DEFAULT_CACHE_DIR):
        self.base_dir = resolve_artifact_path(
            base_dir
        )
        self.data_dir = os.path.join(
            self.base_dir,
            "data",
        )
        self.weight_dir = os.path.join(
            self.base_dir,
            "weights",
        )
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.weight_dir, exist_ok=True)

    def get_data_cache(self, config_dict):
        """
        Load a cached array collection.

        Corrupted or incomplete cache entries are removed and treated as cache
        misses, allowing the calling pipeline to regenerate them.
        """
        uid = _generate_config_hash(
            config_dict
        )

        data_path = os.path.join(
            self.data_dir,
            f"{uid}.npz",
        )

        metadata_path = os.path.join(
            self.data_dir,
            f"{uid}.json",
        )

        if not os.path.exists(data_path):
            return None, uid

        try:
            # The context manager is important on Windows because it closes
            # the underlying ZIP file before this method returns.
            with np.load(
                data_path,
                allow_pickle=True,
            ) as cached_data:
                arrays = {
                    key: cached_data[key]
                    for key in cached_data.files
                }

        except (
            OSError,
            ValueError,
            EOFError,
            pickle.UnpicklingError,
            zipfile.BadZipFile,
        ) as error:
            print(
                "[WARN] Data cache entry "
                f"{uid} is unreadable and will be regenerated: "
                f"{error}"
            )

            _remove_cache_files(
                data_path,
                metadata_path,
            )

            return None, uid

        return arrays, uid

    def save_data_cache(
        self,
        arrays_dict,
        config_dict,
        uid,
    ):
        """
        Save data-cache arrays and metadata using atomic file replacement.
        """
        data_path = os.path.join(
            self.data_dir,
            f"{uid}.npz",
        )

        metadata_path = os.path.join(
            self.data_dir,
            f"{uid}.json",
        )

        def write_metadata(temporary_path):
            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as metadata_file:
                json.dump(
                    config_dict,
                    metadata_file,
                    indent=4,
                    default=str,
                )

        def write_arrays(temporary_path):
            np.savez_compressed(
                temporary_path,
                **arrays_dict,
            )

        # The payload is written last because its existence is used as the
        # cache-hit indicator by get_data_cache().
        _atomic_write_file(
            metadata_path,
            write_metadata,
        )

        _atomic_write_file(
            data_path,
            write_arrays,
        )

    def get_weight_cache(
        self,
        config_dict,
        model,
        device,
    ):
        """
        Load cached model weights and associated training metadata.

        Unreadable or incompatible cache entries are removed and treated
        as cache misses. If loading fails after partially modifying the
        model, its original initialization is restored.
        """
        uid = _generate_config_hash(
            config_dict
        )

        weight_path = os.path.join(
            self.weight_dir,
            f"{uid}.pth",
        )

        metadata_path = os.path.join(
            self.weight_dir,
            f"{uid}.json",
        )

        if not os.path.exists(weight_path):
            return None, uid

        # Preserve the model's initial state because load_state_dict()
        # can partially modify a model before reporting an error.
        original_model_state = copy.deepcopy(
            model.state_dict()
        )

        try:
            cached_state = torch.load(
                weight_path,
                map_location=device,
            )

            model.load_state_dict(
                cached_state
            )

        except (
            OSError,
            EOFError,
            RuntimeError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
        ) as error:
            model.load_state_dict(
                original_model_state
            )

            print(
                "[WARN] Weight cache entry "
                f"{uid} is unreadable or incompatible and "
                f"will be regenerated: {error}"
            )

            _remove_cache_files(
                weight_path,
                metadata_path,
            )

            return None, uid

        # Use the configured maximum epoch count as a fallback for older
        # cache entries that do not contain actual_epochs metadata.
        actual_epochs = config_dict.get(
            "epochs"
        )

        if os.path.exists(metadata_path):
            try:
                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as metadata_file:
                    cache_metadata = json.load(
                        metadata_file
                    )

                actual_epochs = cache_metadata.get(
                    "actual_epochs",
                    actual_epochs,
                )

            except (
                OSError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                print(
                    "[WARN] Could not read weight-cache "
                    f"metadata for hash {uid}: {error}. "
                    "Using the configured epoch count."
                )

        if actual_epochs is not None:
            try:
                model.actual_epochs = int(
                    actual_epochs
                )
            except (TypeError, ValueError):
                model.actual_epochs = config_dict.get(
                    "epochs"
                )

        return model, uid

    def save_weight_cache(
        self,
        model,
        config_dict,
        uid,
    ):
        """
        Save model weights and training metadata atomically.

        The configuration is copied before actual_epochs is added, so the
        dictionary used to generate the cache identity remains unchanged.
        """
        weight_path = os.path.join(
            self.weight_dir,
            f"{uid}.pth",
        )

        metadata_path = os.path.join(
            self.weight_dir,
            f"{uid}.json",
        )

        cache_metadata = dict(
            config_dict
        )

        actual_epochs = getattr(
            model,
            "actual_epochs",
            config_dict.get("epochs"),
        )

        if actual_epochs is not None:
            try:
                actual_epochs = int(
                    actual_epochs
                )
            except (TypeError, ValueError):
                pass

            cache_metadata[
                "actual_epochs"
            ] = actual_epochs

        def write_metadata(temporary_path):
            with open(
                temporary_path,
                "w",
                encoding="utf-8",
            ) as metadata_file:
                json.dump(
                    cache_metadata,
                    metadata_file,
                    indent=4,
                    default=str,
                )

        def write_weights(temporary_path):
            torch.save(
                model.state_dict(),
                temporary_path,
            )

        # Metadata is written first. The weight file is written last
        # because its existence indicates that the cache entry is ready.
        _atomic_write_file(
            metadata_path,
            write_metadata,
        )

        _atomic_write_file(
            weight_path,
            write_weights,
        )

