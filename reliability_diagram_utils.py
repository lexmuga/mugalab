"""
MUGA LAB Reliability Diagram Visualization Utilities
----------------------------------------------------

Companion module for calibration_metrics.py.
Provides side-by-side and comparative reliability diagram plotting,
and integrates optional MLflow logging for reproducible calibration
visualization in MUGA LAB experiments.

Functions:
    - plot_reliability
    - compare_reliability
    - log_reliability_to_mlflow

Author: MUGA LAB Prompt Engineer
License: MIT
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import mlflow
from typing import Optional, Tuple

from mugalab.calibration_metrics import (
    expected_calibration_error,
    miscalibration_score,
    reliability_diagram,
    _to_numpy,
    softmax
)


# ============================================================
# 🔹 1. Basic Reliability Plot
# ============================================================

def plot_reliability(
    logits,
    labels,
    n_bins: int = 15,
    figsize: Tuple[int, int] = (6, 6),
    title: str = "Reliability Diagram",
    color: str = "C0",
    show_ece: bool = True,
    savepath: Optional[str] = None,
):
    """
    Plot a single reliability diagram with ECE and MCS annotations.

    Parameters
    ----------
    logits : torch.Tensor or np.ndarray
    labels : torch.Tensor or np.ndarray
    n_bins : int
    figsize : tuple
    title : str
    color : str
    show_ece : bool
    savepath : str or None

    Returns
    -------
    (ece, mcs)
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

    ece = expected_calibration_error(logits, labels, n_bins)
    mcs = miscalibration_score(logits, labels, n_bins)

    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(
        bin_confs,
        np.array(bin_accs) - np.array(bin_confs),
        width=1.0 / n_bins,
        alpha=0.6,
        color=color,
        edgecolor="k",
        label="Gap (Accuracy - Confidence)",
    )
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    if show_ece:
        plt.text(
            0.05,
            0.85,
            f"ECE={ece:.3f}\nMCS={mcs:.3f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="gray"),
        )
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    return ece, mcs


# ============================================================
# 🔹 2. Comparison Plot: Pre vs Post Calibration
# ============================================================

def compare_reliability(
    logits_uncal,
    logits_cal,
    labels,
    n_bins: int = 15,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Pre vs Post Calibration",
    savepath: Optional[str] = None,
):
    """
    Generate side-by-side reliability diagrams before and after calibration.

    Parameters
    ----------
    logits_uncal : model outputs before calibration
    logits_cal : model outputs after temperature scaling/distillation
    labels : true labels
    """
    ece_before = expected_calibration_error(logits_uncal, labels, n_bins)
    mcs_before = miscalibration_score(logits_uncal, labels, n_bins)
    ece_after = expected_calibration_error(logits_cal, labels, n_bins)
    mcs_after = miscalibration_score(logits_cal, labels, n_bins)

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Uncalibrated
    for ax, logits, ece, mcs, title_suffix, color in [
        (axs[0], logits_uncal, ece_before, mcs_before, "Uncalibrated", "C1"),
        (axs[1], logits_cal, ece_after, mcs_after, "Calibrated", "C2"),
    ]:
        probs = softmax(_to_numpy(logits))
        labels_np = _to_numpy(labels)
        confs = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accs, bin_confs = [], []

        for i in range(n_bins):
            mask = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
            if np.any(mask):
                bin_accs.append(np.mean(preds[mask] == labels_np[mask]))
                bin_confs.append(np.mean(confs[mask]))
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.bar(
            bin_confs,
            np.array(bin_accs) - np.array(bin_confs),
            width=1.0 / n_bins,
            alpha=0.6,
            color=color,
            edgecolor="k",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{title_suffix}\nECE={ece:.3f}, MCS={mcs:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()

    return {
        "ece_before": ece_before,
        "mcs_before": mcs_before,
        "ece_after": ece_after,
        "mcs_after": mcs_after,
    }


# ============================================================
# 🔹 3. MLflow Logging Integration
# ============================================================

def log_reliability_to_mlflow(
    logits_uncal,
    logits_cal,
    labels,
    n_bins: int = 15,
    fig_title: str = "Calibration Comparison",
    artifact_name: str = "reliability_diagram.png",
):
    """
    Generate a comparison diagram and log it to MLflow as an artifact.
    """
    import tempfile

    metrics = compare_reliability(
        logits_uncal,
        logits_cal,
        labels,
        n_bins=n_bins,
        figsize=(12, 5),
        title=fig_title,
        savepath=None,
    )

    # Log metrics to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Save plot temporarily
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        compare_reliability(
            logits_uncal,
            logits_cal,
            labels,
            n_bins=n_bins,
            figsize=(12, 5),
            title=fig_title,
            savepath=tmp_path,
        )
        mlflow.log_artifact(tmp_path, artifact_path="figures")

    print(f"[MLflow] Logged reliability comparison and metrics: {artifact_name}")
    return metrics
