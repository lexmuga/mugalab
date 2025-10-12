#!/usr/bin/env python3
"""
mugalab/undergraduate_research/experiments/04_cross_architecture_eval.py
------------------------------------------------------------

Cross-architecture evaluation for MUGA LAB undergraduate research.

This script:
    - Loads multiple trained models (e.g., baseline, calibrated, distilled).
    - Evaluates each on the same validation dataset.
    - Computes accuracy, ECE, MCS, and NLL.
    - Logs all results to MLflow 3.0.
    - Exports a LaTeX-ready comparative table.

Usage:
    python 04_cross_architecture_eval.py \
        --train ../../data/train.csv \
        --target label \
        --models runs:/<RUN_ID1>/model runs:/<RUN_ID2>/model runs:/<RUN_ID3>/model \
        --labels Baseline Calibrated Distilled
"""

import argparse
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from mugalab.mlp_tuner_tabular_mlflow import load_tabular_csv, get_device
from mugalab.calibration_metrics import (
    expected_calibration_error,
    miscalibration_score,
    negative_log_likelihood,
)
from mugalab.reliability_diagram_utils import log_reliability_to_mlflow


# ============================================================
# Evaluation Function
# ============================================================

@torch.no_grad()
def evaluate_model(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    """Evaluate a model on validation data.

    Args:
        model: Trained PyTorch model.
        X: Validation features tensor.
        y: Validation labels tensor.

    Returns:
        Dictionary containing accuracy and calibration metrics.
    """
    device = get_device()
    model.to(device).eval()
    X, y = X.to(device), y.to(device)
    logits = model(X)

    accuracy = (logits.argmax(dim=1) == y).float().mean().item()
    ece = expected_calibration_error(logits, y)
    mcs = miscalibration_score(logits, y)
    nll = negative_log_likelihood(logits, y)

    return {
        "accuracy": accuracy,
        "ECE": ece,
        "MCS": mcs,
        "NLL": nll,
        "logits": logits,
    }


# ============================================================
# Cross-Architecture Evaluation
# ============================================================

def run_cross_architecture_evaluation(
    model_uris: list,
    model_labels: list,
    train_path: str,
    target_col: str,
    mlflow_uri: str = "mlruns",
    experiment_name: str = "Cross_Architecture_Evaluation",
    seed: int = 42
) -> None:
    """Compare multiple architectures in terms of accuracy and calibration.

    Args:
        model_uris: List of MLflow model URIs or local paths.
        model_labels: List of human-readable labels for models.
        train_path: Path to the dataset CSV.
        target_col: Name of the target column.
        mlflow_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        seed: Random seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    (X_train, y_train), (X_val, y_val), task = load_tabular_csv(train_path, target_col, seed)

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    results = []
    logits_collection = {}

    with mlflow.start_run(run_name="Cross_Architecture_Comparison"):
        for uri, label in zip(model_uris, model_labels):
            print(f"Evaluating model: {label}")
            model = mlflow.pytorch.load_model(uri)
            metrics = evaluate_model(model, X_val, y_val)
            results.append({
                "Model": label,
                "Accuracy": metrics["accuracy"],
                "ECE": metrics["ECE"],
                "MCS": metrics["MCS"],
                "NLL": metrics["NLL"],
            })
            logits_collection[label] = metrics["logits"]

            # Log metrics to MLflow
            mlflow.log_metrics({
                f"{label}_accuracy": metrics["accuracy"],
                f"{label}_ECE": metrics["ECE"],
                f"{label}_MCS": metrics["MCS"],
                f"{label}_NLL": metrics["NLL"],
            })

        # Generate comparative reliability diagram between first and last models
        if len(logits_collection) >= 2:
            labels_list = list(logits_collection.keys())
            log_reliability_to_mlflow(
                logits_collection[labels_list[0]],
                logits_collection[labels_list[-1]],
                y_val,
                fig_title=f"{labels_list[0]} vs {labels_list[-1]} Reliability Comparison"
            )

        # Export results as LaTeX table
        df_results = pd.DataFrame(results)
        latex_table = df_results.to_latex(
            float_format="%.4f",
            caption="Cross-Architecture Calibration and Accuracy Comparison",
            label="tab:cross_architecture_eval",
            index=False
        )

        with open("../../reports/tables/cross_architecture_eval_table.tex", "w") as f:
            f.write(latex_table)

        mlflow.log_artifact("../../reports/tables/cross_architecture_eval_table.tex", artifact_path="tables")
        print("Exported LaTeX comparison table and logged results to MLflow.")


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    """Command-line entry point for cross-architecture evaluation."""
    parser = argparse.ArgumentParser(description="Cross-Architecture Calibration Evaluation")
    parser.add_argument("--models", nargs="+", required=True, help="List of MLflow model URIs or local paths.")
    parser.add_argument("--labels", nargs="+", required=True, help="List of model labels for reporting.")
    parser.add_argument("--train", type=str, required=True, help="Path to dataset CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Target column name.")
    parser.add_argument("--mlflow_uri", type=str, default="../../results/mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment", type=str, default="Cross_Architecture_Evaluation", help="MLflow experiment name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if len(args.models) != len(args.labels):
        raise ValueError("The number of models must match the number of labels.")

    run_cross_architecture_evaluation(
        model_uris=args.models,
        model_labels=args.labels,
        train_path=args.train,
        target_col=args.target,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        seed=args.seed
    )


if __name__ == "__main__":
    main()