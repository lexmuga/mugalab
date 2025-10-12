#!/usr/bin/env python3
"""
mugalab/undergraduate_research/experiments/05_calibration_summary.py
------------------------------------------------------------

Aggregates calibration and accuracy metrics across experiments.

This script:
    - Fetches results from MLflow tracking.
    - Aggregates metrics from Baseline, Calibrated, Distilled, and Cross-Architecture runs.
    - Computes mean, standard deviation, and summary statistics.
    - Exports CSV and LaTeX tables for thesis reporting.

Usage:
    python 05_calibration_summary.py --mlflow_uri ../../results/mlruns --output_dir ../../reports/summary
"""

import argparse
import os
import pandas as pd
import mlflow


# ============================================================
# Data Extraction Utilities
# ============================================================

def fetch_experiment_data(experiment_name: str, tracking_uri: str) -> pd.DataFrame:
    """Fetch all runs for a given MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_uri: Path or URI to MLflow tracking server.

    Returns:
        DataFrame containing run parameters and metrics.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow tracking URI: {tracking_uri}")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    df = df[["run_id", "start_time", "metrics.val_loss", "metrics.val_acc", "params.search_method"]].copy()
    df.rename(columns={
        "metrics.val_loss": "val_loss",
        "metrics.val_acc": "val_acc",
        "params.search_method": "search_method"
    }, inplace=True)
    df.sort_values(by="val_loss", inplace=True)
    return df


def summarize_metrics(dfs: dict) -> pd.DataFrame:
    """Compute mean and standard deviation for calibration metrics.

    Args:
        dfs: Dictionary of experiment names to DataFrames.

    Returns:
        Combined summary DataFrame.
    """
    summary_rows = []
    for name, df in dfs.items():
        mean_loss = df["val_loss"].mean()
        std_loss = df["val_loss"].std()
        mean_acc = df["val_acc"].mean()
        std_acc = df["val_acc"].std()
        summary_rows.append({
            "Experiment": name,
            "Mean Val Loss": mean_loss,
            "Std Val Loss": std_loss,
            "Mean Val Accuracy": mean_acc,
            "Std Val Accuracy": std_acc,
            "Num Runs": len(df)
        })
    return pd.DataFrame(summary_rows)


# ============================================================
# Summary Export
# ============================================================

def export_summary(summary_df: pd.DataFrame, output_dir: str) -> None:
    """Export summary results to CSV and LaTeX formats."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "calibration_summary.csv")
    tex_path = os.path.join(output_dir, "calibration_summary.tex")

    summary_df.to_csv(csv_path, index=False)
    summary_df.to_latex(
        tex_path,
        float_format="%.4f",
        caption="Calibration and Accuracy Summary Across Experiments",
        label="tab:calibration_summary",
        index=False
    )
    print(f"Exported calibration summary to: {csv_path}")
    print(f"Exported LaTeX table to: {tex_path}")


# ============================================================
# Main Runner
# ============================================================

def run_calibration_summary(mlflow_uri: str, output_dir: str) -> None:
    """Aggregate experiment results and produce summary tables."""
    experiments = [
        "Baseline_Tuning",
        "Temperature_Scaling",
        "Distillation_Experiment",
        "Cross_Architecture_Evaluation"
    ]

    dfs = {}
    for exp in experiments:
        try:
            df = fetch_experiment_data(exp, mlflow_uri)
            dfs[exp] = df
        except ValueError as e:
            print(str(e))

    summary_df = summarize_metrics(dfs)
    export_summary(summary_df, output_dir)


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    """Command-line entry point for calibration summary."""
    parser = argparse.ArgumentParser(description="Aggregate calibration and accuracy results from MLflow.")
    parser.add_argument("--mlflow_uri", type=str, default="../../results/mlruns", help="MLflow tracking URI.")
    parser.add_argument("--output_dir", type=str, default="../../reports/summary", help="Output directory for reports.")
    args = parser.parse_args()

    run_calibration_summary(mlflow_uri=args.mlflow_uri, output_dir=args.output_dir)


if __name__ == "__main__":
    main()