#!/usr/bin/env python3
"""
MLP Hyperparameter Tuning Module (Optuna + DEHB + MLflow 3.0)
--------------------------------------------------------------

Implements a reproducible workflow for tuning Multilayer Perceptrons
on tabular datasets. Supports both Optuna and DEHB search strategies,
with full experiment tracking via MLflow 3.0.

This module is compatible with Apple Silicon (MPS), CUDA GPUs,
and CPU environments.
"""

import argparse
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import Trial
from dehb import DEHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import mlflow
import mlflow.pytorch


# ============================================================
# Device selection
# ============================================================

def get_device() -> torch.device:
    """Select the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: Selected compute device.
    """
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using Apple MPS backend")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


# ============================================================
# Data utilities
# ============================================================

def detect_task(y: np.ndarray) -> str:
    """Determine whether the task is classification or regression."""
    unique = np.unique(y)
    if y.dtype.kind in "ifu" and len(unique) > 20 and np.all(~np.isin(y, [0, 1])):
        return "regression"
    return "classification"


def load_tabular_csv(
    train_path: str, target_col: str, seed: int = 42
) -> Tuple[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor],
           str]:
    """Load tabular CSV data and prepare PyTorch tensors.

    Args:
        train_path: Path to the CSV file.
        target_col: Name of the target column.
        seed: Random seed.

    Returns:
        Tuple of (train tensors, validation tensors, task type).
    """
    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    task = detect_task(y)

    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed,
        stratify=y if task == "classification" else None
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long if task == "classification" else torch.float32)

    return (X_train, y_train), (X_val, y_val), task


# ============================================================
# Model definition
# ============================================================

ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU
}


class MLP(nn.Module):
    """Configurable Multilayer Perceptron model."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_units: list,
        activation: str,
        dropout: float,
        batch_norm: bool
    ):
        super().__init__()
        layers = []
        act_fn = ACTS[activation]
        prev_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    l2_weight: float
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        if l2_weight > 0:
            l2 = sum((p ** 2).sum() for p in model.parameters())
            loss += l2_weight * l2
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    task: str
) -> Tuple[float, float]:
    """Evaluate model performance on validation data."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb)
        loss = criterion(preds, yb)
        total_loss += loss.item() * xb.size(0)
        if task == "classification":
            pred = preds.argmax(dim=1)
            correct += (pred == yb).sum().item()
        n += xb.size(0)
    accuracy = correct / n if task == "classification" else -total_loss / n
    return total_loss / n, accuracy


def make_optimizer(
    params,
    name: str,
    lr: float,
    momentum: float,
    beta1: float,
    beta2: float,
    wd: float
) -> torch.optim.Optimizer:
    """Instantiate an optimizer by name."""
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2), weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# Configuration space for hyperparameter optimization
# ============================================================

def configspace(max_layers: int = 5) -> CS.ConfigurationSpace:
    """Define hyperparameter search space."""
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameters([
        CSH.UniformIntegerHyperparameter("n_layers", 1, max_layers),
        *[CSH.UniformIntegerHyperparameter(f"n_units_l{i}", 32, 1024, log=True)
          for i in range(max_layers)],
        CSH.CategoricalHyperparameter("activation", ["relu", "tanh", "gelu", "leaky_relu"]),
        CSH.CategoricalHyperparameter("optimizer", ["adam", "sgd", "rmsprop"]),
        CSH.UniformFloatHyperparameter("learning_rate", 1e-5, 1e-2, log=True),
        CSH.CategoricalHyperparameter("batch_size", [16, 32, 64, 128, 256]),
        CSH.UniformFloatHyperparameter("momentum", 0.5, 0.99),
        CSH.UniformFloatHyperparameter("beta_1", 0.8, 0.99),
        CSH.UniformFloatHyperparameter("beta_2", 0.9, 0.9999),
        CSH.UniformFloatHyperparameter("dropout_rate", 0.0, 0.5),
        CSH.UniformFloatHyperparameter("l2_penalty", 1e-6, 1e-2, log=True),
        CSH.CategoricalHyperparameter("batch_norm", [True, False]),
        CSH.UniformIntegerHyperparameter("patience", 5, 20),
    ])
    return cs


# ============================================================
# Objective function with MLflow logging
# ============================================================

def objective_with_logging(
    config: Dict[str, Any],
    budget_epochs: int,
    data: Tuple[Tuple[torch.Tensor, torch.Tensor],
                Tuple[torch.Tensor, torch.Tensor]],
    task: str,
    run_name: str = None
) -> float:
    """Objective function for Optuna and DEHB with MLflow tracking."""
    (X_train, y_train), (X_val, y_val) = data
    device = get_device()
    bs = int(config["batch_size"])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs)

    input_dim = X_train.shape[1]
    output_dim = int(y_train.max().item() + 1) if task == "classification" else 1
    hidden = [int(config[f"n_units_l{i}"]) for i in range(int(config["n_layers"]))]

    model = MLP(
        input_dim, output_dim, hidden,
        config["activation"], config["dropout_rate"], config["batch_norm"]
    ).to(device)

    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
    optimizer = make_optimizer(
        model.parameters(),
        config["optimizer"],
        lr=config["learning_rate"],
        momentum=config["momentum"],
        beta1=config["beta_1"],
        beta2=config["beta_2"],
        wd=config["l2_penalty"]
    )

    patience = int(config["patience"])
    best_val, counter = float("inf"), 0

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(config)

        for epoch in range(budget_epochs):
            train_one_epoch(model, train_loader, device, optimizer, criterion, 0.0)
            val_loss, val_acc = eval_loss(model, val_loader, device, criterion, task)
            mlflow.log_metrics({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                counter = 0
                mlflow.pytorch.log_model(model, artifact_path="model")
            else:
                counter += 1
            if counter >= patience:
                break

    return best_val


# ============================================================
# Main entry point
# ============================================================

def main():
    """Run MLP hyperparameter tuning with Optuna or DEHB."""
    parser = argparse.ArgumentParser(description="MLP Hyperparameter Tuner")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--search", type=str, choices=["optuna", "dehb"], default="optuna")
    parser.add_argument("--time", type=int, default=300)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlflow_uri", type=str, default="mlruns")
    parser.add_argument("--experiment", type=str, default="MLP_Tuning")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    (Xtr, ytr), (Xval, yval), task = load_tabular_csv(args.train, args.target, args.seed)
    data = ((Xtr, ytr), (Xval, yval))
    cs = configspace()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"{args.search.upper()}_Run"):
        mlflow.log_params({
            "search_method": args.search,
            "seed": args.seed,
            "task": task,
        })

        if args.search == "optuna":
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

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(optuna_objective, n_trials=args.trials)
            mlflow.log_params(study.best_params)
            mlflow.log_metrics({"best_val_loss": study.best_value})
            print(f"Optuna best validation loss: {study.best_value:.4f}")

        elif args.search == "dehb":
            def f(cfg, budget):
                return objective_with_logging(cfg.get_dictionary(), int(budget), data, task, run_name="DEHB_trial")

            dehb = DEHB(f=f, cs=cs, min_fidelity=5, max_fidelity=200, eta=3, random_state=args.seed)
            dehb.run(total_cost=args.time)
            mlflow.log_params(dehb.inc_config)
            mlflow.log_metrics({"best_val_loss": dehb.inc_score})
            print(f"DEHB best validation loss: {dehb.inc_score:.4f}")


if __name__ == "__main__":
    main()
