#!/usr/bin/env python3
"""
mugalab/undergraduate_research/experiments/02_temperature_scaling.py
------------------------------------------------------------

Performs post-hoc temperature scaling calibration on a trained MLP.

This script:
    - Loads a tuned model (from MLflow or local path).
    - Optimizes temperature T* using Optuna.
    - Evaluates calibration metrics (ECE, MCS, NLL).
    - Logs results, figures, and best T* to MLflow 3.0.

Usage:
    python 02_temperature_scaling.py --model_uri runs:/<RUN_ID>/model --train ../../data/train.csv --target label
"""

import argparse
import numpy as np
import torch
import mlflow
import optuna

from torch import nn

from mugalab.mlp_tuner_tabular_mlflow import load_tabular_csv, get_device
from mugalab.calibration_metrics import (
    expected_calibration_error,
    miscalibration_score,
    negative_log_likelihood,
)
from mugalab.reliability_diagram_utils import log_reliability_to_mlflow


# ============================================================
# Temperature Scaling Module
# ============================================================

class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling calibration module."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature.clamp(min=1e-6)

    def set_temperature(self, T: float):
        """Set temperature to a specific scalar value."""
        self.temperature.data = torch.tensor([T], dtype=torch.float32)


# ============================================================
# Temperature Optimization
# ============================================================

def optimize_temperature(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    task: str,
    n_trials: int = 50
) -> float:
    """Find optimal temperature T* minimizing NLL on validation data."""
    device = get_device()
    model.eval()

    X_val, y_val = X_val.to(device), y_val.to(device)

    def objective(trial: optuna.Trial) -> float:
        T = trial.suggest_float("temperature", 0.5, 5.0)
        scaled_model = TemperatureScaling(model)
        scaled_model.set_temperature(T)
        scaled_model.to(device)
        with torch.no_grad():
            logits = scaled_model(X_val)
            loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
            loss = loss_fn(logits, y_val)
        return float(loss.item())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_T = study.best_params["temperature"]
    print(f"Optimal temperature T*: {best_T:.4f}")
    return best_T


# ============================================================
# Calibration Evaluation
# ============================================================

def evaluate_calibration(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    task: str,
    T: float
) -> dict:
    """Evaluate calibration metrics before and after temperature scaling."""
    device = get_device()
    model.to(device).eval()
    X_val, y_val = X_val.to(device), y_val.to(device)

    with torch.no_grad():
        logits_uncal = model(X_val)
        scaled_model = TemperatureScaling(model)
        scaled_model.set_temperature(T)
        scaled_model.to(device)
        logits_cal = scaled_model(X_val)

    metrics = {
        "ECE_before": expected_calibration_error(logits_uncal, y_val),
        "ECE_after": expected_calibration_error(logits_cal, y_val),
        "MCS_before": miscalibration_score(logits_uncal, y_val),
        "MCS_after": miscalibration_score(logits_cal, y_val),
        "NLL_before": negative_log_likelihood(logits_uncal, y_val),
        "NLL_after": negative_log_likelihood(logits_cal, y_val),
    }
    return metrics, logits_uncal, logits_cal


# ============================================================
# Experiment Runner
# ============================================================

def run_temperature_scaling(
    model_uri: str,
    train_path: str,
    target_col: str,
    mlflow_uri: str = "mlruns",
    experiment_name: str = "Temperature_Scaling",
    trials: int = 50,
    seed: int = 42
) -> None:
    """Run temperature scaling experiment and log to MLflow.

    Args:
        model_uri: MLflow model URI (e.g., runs:/<RUN_ID>/model).
        train_path: Path to the CSV training data.
        target_col: Target column name.
        mlflow_uri: MLflow tracking URI.
        experiment_name: Name of the MLflow experiment.
        trials: Number of Optuna trials for temperature search.
        seed: Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    (X_train, y_train), (X_val, y_val), task = load_tabular_csv(train_path, target_col, seed)
    device = get_device()

    # Load model
    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()

    # Optimize temperature
    best_T = optimize_temperature(model, X_val, y_val, task, n_trials=trials)

    # Evaluate calibration metrics
    metrics, logits_uncal, logits_cal = evaluate_calibration(model, X_val, y_val, task, best_T)

    # Log everything to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Temperature_Scaling"):
        mlflow.log_param("optimal_temperature", best_T)
        mlflow.log_metrics(metrics)
        log_reliability_to_mlflow(logits_uncal, logits_cal, y_val)
        print("Logged calibration metrics and reliability diagram to MLflow.")


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    """Command-line entry point for temperature scaling calibration."""
    parser = argparse.ArgumentParser(description="Post-hoc Temperature Scaling Calibration")
    parser.add_argument("--model_uri", type=str, required=True, help="MLflow model URI or local path.")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Target column name.")
    parser.add_argument("--mlflow_uri", type=str, default="../../results/mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment", type=str, default="Temperature_Scaling", help="MLflow experiment name.")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_temperature_scaling(
        model_uri=args.model_uri,
        train_path=args.train,
        target_col=args.target,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        trials=args.trials,
        seed=args.seed
    )


if __name__ == "__main__":
    main()