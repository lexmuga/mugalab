"""
MUGA LAB Calibration Metrics Module
-----------------------------------
Implements standard and advanced calibration metrics for evaluating
non-image neural networks (classification focus).

Metrics:
    - Expected Calibration Error (ECE)
    - Classwise ECE (per-class calibration)
    - Maximum Calibration Error (MCE)
    - Negative Log-Likelihood (NLL)
    - Miscalibration Score (MCS)
    - Reliability Diagram plotting utility

Author: MUGA LAB Prompt Engineer
License: MIT
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


# ============================================================
# 🔹 1. Helper Functions
# ============================================================

def _to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def softmax(x, axis=1):
    """Numerically stable softmax."""
    x = np.array(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================
# 🔹 2. Expected Calibration Error (ECE)
# ============================================================

def expected_calibration_error(
    logits, labels, n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    logits : np.ndarray or torch.Tensor
        Model logits (n_samples, n_classes)
    labels : np.ndarray or torch.Tensor
        True labels (n_samples,)
    n_bins : int
        Number of bins to partition confidence scores

    Returns
    -------
    float : ECE value in [0,1]
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            acc_bin = np.mean(predictions[mask] == labels[mask])
            conf_bin = np.mean(confidences[mask])
            ece += np.abs(acc_bin - conf_bin) * np.sum(mask)

    return ece / len(labels)


# ============================================================
# 🔹 3. Classwise ECE
# ============================================================

def classwise_ece(
    logits, labels, n_bins: int = 15
) -> Dict[int, float]:
    """
    Compute classwise Expected Calibration Error (ECE per class).

    Returns
    -------
    dict : {class_index: ece_value}
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels)
    n_classes = probs.shape[1]
    results = {}

    for c in range(n_classes):
        mask = labels == c
        if np.sum(mask) == 0:
            continue
        ece_c = expected_calibration_error(
            logits[mask], labels[mask], n_bins=n_bins
        )
        results[c] = ece_c
    return results


# ============================================================
# 🔹 4. Maximum Calibration Error (MCE)
# ============================================================

def maximum_calibration_error(
    logits, labels, n_bins: int = 15
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    MCE = max_m |acc(B_m) - conf(B_m)|
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    cal_errors = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            acc_bin = np.mean(predictions[mask] == labels[mask])
            conf_bin = np.mean(confidences[mask])
            cal_errors.append(np.abs(acc_bin - conf_bin))
    return np.max(cal_errors) if cal_errors else 0.0


# ============================================================
# 🔹 5. Negative Log-Likelihood (NLL)
# ============================================================

def negative_log_likelihood(logits, labels) -> float:
    """
    Compute mean negative log-likelihood.
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels).astype(int)
    eps = 1e-12
    log_probs = -np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0))
    return np.mean(log_probs)


# ============================================================
# 🔹 6. Miscalibration Score (MCS)
# ============================================================

def miscalibration_score(
    logits, labels, n_bins: int = 15
) -> float:
    """
    Compute Miscalibration Score (MCS).
    A variant capturing *both* over- and under-confidence by
    squaring deviations instead of taking absolute value.

    MCS = sum_m (|B_m| / n) * (acc(B_m) - conf(B_m))^2
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mcs = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            acc_bin = np.mean(predictions[mask] == labels[mask])
            conf_bin = np.mean(confidences[mask])
            mcs += ((acc_bin - conf_bin) ** 2) * np.sum(mask)

    return mcs / len(labels)


# ============================================================
# 🔹 7. Reliability Diagram (Visualization)
# ============================================================

def reliability_diagram(
    logits, labels, n_bins: int = 15, figsize: Tuple[int, int] = (6, 6),
    title: str = "Reliability Diagram", savepath: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot a reliability diagram for visualization of calibration.

    Returns:
        bin_confidences, bin_accuracies
    """
    probs = softmax(_to_numpy(logits))
    labels = _to_numpy(labels)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs = [], []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if np.any(mask):
            bin_accs.append(np.mean(predictions[mask] == labels[mask]))
            bin_confs.append(np.mean(confidences[mask]))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(bin_confs, np.array(bin_accs) - np.array(bin_confs),
            width=1.0 / n_bins, alpha=0.5, color="C0", edgecolor="k",
            label="Gap (Accuracy - Confidence)")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    return np.array(bin_confs), np.array(bin_accs)
