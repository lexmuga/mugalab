#!/usr/bin/env python3
"""
mugalab/undergraduate_research/utils/seed_sensitivity_utils.py
------------------------------------------------------------

Multi-seed evaluation utilities for reproducibility analysis.

This module enables:
    - Multi-seed repeated execution of model evaluation or tuning objectives.
    - Computation of mean ± standard deviation for calibration and accuracy metrics.
    - Logging of variance-aware results to MLflow 3.0.

Intended for use with experiments:
    01_baseline_tuning.py
    02_temperature_scaling.py
    03_distillation_experiment.py
    04_cross_architecture_eval.py

Example:
    from seed_sensitivity_utils import evaluate_across_seeds

    results = evaluate_across_seeds(
        func=objective_with_logging,
        seeds=[0, 21, 42, 84, 126],
        config=cfg,
        data=data,
        task="classification",
        mlflow_experiment="Seed_Sensitivity"
    )
"""

import numpy as np
import pandas as pd
import torch
import mlflow
from typing import Callable, Dict, Any, List


# ============================================================
# Multi-Seed Evaluation Wrapper
# ============================================================

def evaluate_across_seeds(
    func: Callable[..., float],
    seeds: List[int],
    mlflow_experiment: str,
    mlflow_uri: str = "mlruns",
    **kwargs
) -> Dict[str, Any]:
    """Evaluate a function across multiple random seeds and log statistics.

    Args:
        func: Objective or evaluation function to run (returns a scalar loss/metric).
        seeds: List of random seeds to test.
        mlflow_experiment: Name of MLflow experiment for tracking.
        mlflow_uri: Path or URI for MLflow tracking.
        **kwargs: Additional keyword arguments passed to `func`.

    Returns:
        Dictionary containing mean, std, and per-seed results.
    """
    results = []

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="Seed_Sensitivity_Evaluation"):
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            print(f"Running evaluation for seed: {seed}")
            result = func(**kwargs)
            results.append(result)
            mlflow.log_metric(f"val_metric_seed_{seed}", result)

        mean_result = float(np.mean(results))
        std_result = float(np.std(results))

        mlflow.log_metrics({
            "mean_val_metric": mean_result,
            "std_val_metric": std_result
        })

        summary = {
            "seeds": seeds,
            "results": results,
            "mean": mean_result,
            "std": std_result
        }

        print(f"Mean: {mean_result:.4f}, Std: {std_result:.4f}")
        return summary


# ============================================================
# Multi-Metric Evaluation Utility
# ============================================================

def evaluate_metrics_across_seeds(
    func: Callable[..., Dict[str, float]],
    seeds: List[int],
    metric_keys: List[str],
    mlflow_experiment: str,
    mlflow_uri: str = "mlruns",
    **kwargs
) -> pd.DataFrame:
    """Run a multi-metric evaluation across seeds (e.g., ECE, MCS, NLL).

    Args:
        func: Function returning a dictionary of metrics.
        seeds: List of random seeds to evaluate.
        metric_keys: Keys in the metric dictionary to aggregate.
        mlflow_experiment: MLflow experiment name.
        mlflow_uri: MLflow tracking URI.
        **kwargs: Additional parameters passed to `func`.

    Returns:
        DataFrame containing per-seed metrics and mean ± std summary.
    """
    records = []

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name="Seed_Sensitivity_Metrics"):
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            print(f"Evaluating metrics for seed: {seed}")
            metrics = func(**kwargs)
            metrics["seed"] = seed
            records.append(metrics)

            for key, value in metrics.items():
                if key != "seed":
                    mlflow.log_metric(f"{key}_seed_{seed}", value)

        df = pd.DataFrame(records)
        summary = df[metric_keys].agg(["mean", "std"]).reset_index()
        summary.rename(columns={"index": "Statistic"}, inplace=True)

        for key in metric_keys:
            mlflow.log_metric(f"{key}_mean", summary.loc[summary["Statistic"] == "mean", key].values[0])
            mlflow.log_metric(f"{key}_std", summary.loc[summary["Statistic"] == "std", key].values[0])

        print("Multi-seed metric summary:")
        print(summary)

        return df


# ============================================================
# CSV Export Utility
# ============================================================

def export_seed_results(df: pd.DataFrame, output_path: str) -> None:
    """Export multi-seed evaluation results to CSV.

    Args:
        df: DataFrame containing per-seed results.
        output_path: Path to CSV output file.
    """
    df.to_csv(output_path, index=False)
    print(f"Exported multi-seed results to: {output_path}")
