#!/usr/bin/env python3

"""
mugalab/undergraduate_research/experiments/01_baseline_tuning.py

------------------------------------------------------------

Baseline MLP hyperparameter tuning for tabular datasets.

This script:
    - Loads and preprocesses a tabular dataset.
    - Tunes an MLP using Optuna or DEHB.
    - Logs results and best parameters to MLflow 3.0.
    - Saves the best trained model as an MLflow artifact.

Usage:
    python 01_baseline_tuning.py --train ../../data/train.csv --target label --search optuna
"""

import argparse
import numpy as np
import torch
import mlflow

from mugalab.mlp_tuner_tabular_mlflow import (
    load_tabular_csv,
    configspace,
    objective_with_logging,
)
import optuna
from optuna.trial import Trial
from dehb import DEHB
import ConfigSpace.hyperparameters as CSH


# ============================================================
# Device and Reproducibility
# ============================================================

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main Baseline Tuning Function
# ============================================================

def run_baseline_tuning(
    train_path: str,
    target_col: str,
    search_method: str = "optuna",
    mlflow_uri: str = "mlruns",
    experiment_name: str = "Baseline_Tuning",
    trials: int = 30,
    max_time: int = 300,
    seed: int = 42
) -> None:
    """Run baseline MLP tuning and log results to MLflow.

    Args:
        train_path: Path to training CSV file.
        target_col: Name of the target column in the dataset.
        search_method: 'optuna' or 'dehb'.
        mlflow_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        trials: Number of Optuna trials.
        max_time: Time budget for DEHB search.
        seed: Random seed for reproducibility.
    """
    set_random_seeds(seed)

    (X_train, y_train), (X_val, y_val), task = load_tabular_csv(train_path, target_col, seed)
    data = ((X_train, y_train), (X_val, y_val))
    cs = configspace()

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{search_method.upper()}_Baseline"):
        mlflow.log_params({
            "search_method": search_method,
            "seed": seed,
            "task": task,
        })

        if search_method == "optuna":
            def optuna_objective(trial: Trial):
                cfg = {
                    hp.name: (
                        trial.suggest_float(hp.name, hp.lower, hp.upper, log=hp.log)
                        if isinstance(hp, CSH.UniformFloatHyperparameter)
                        else trial.suggest_categorical(hp.name, hp.choices)
                        if isinstance(hp, CSH.CategoricalHyperparameter)
                        else trial.suggest_int(hp.name, hp.lower, hp.upper, log=hp.log)
                    )
                    for hp in cs.get_hyperparameters()
                }
                return objective_with_logging(cfg, 200, data, task, run_name=f"trial_{trial.number}")

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(optuna_objective, n_trials=trials)
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"best_val_loss": study.best_value})
            print(f"Optuna best validation loss: {study.best_value:.4f}")

        elif search_method == "dehb":
            def f(cfg, budget):
                return objective_with_logging(cfg.get_dictionary(), int(budget), data, task, run_name="DEHB_trial")

            dehb = DEHB(f=f, cs=cs, min_fidelity=5, max_fidelity=200, eta=3, random_state=seed)
            dehb.run(total_cost=max_time)
            mlflow.log_params(dehb.inc_config)
            mlflow.log_metrics({"best_val_loss": dehb.inc_score})
            print(f"DEHB best validation loss: {dehb.inc_score:.4f}")

        else:
            raise ValueError("Unsupported search method. Use 'optuna' or 'dehb'.")


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Baseline MLP Tuning Experiment")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Target column name.")
    parser.add_argument("--search", type=str, choices=["optuna", "dehb"], default="optuna")
    parser.add_argument("--mlflow_uri", type=str, default="../../results/mlruns")
    parser.add_argument("--experiment", type=str, default="Baseline_Tuning")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--time", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_baseline_tuning(
        train_path=args.train,
        target_col=args.target,
        search_method=args.search,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        trials=args.trials,
        max_time=args.time,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
