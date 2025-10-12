#!/usr/bin/env python3
"""
mugalab/undergraduate_research/experiments/03_distillation_experiment.py
------------------------------------------------------------

Knowledge distillation experiment using a calibrated teacher model.

This script:
    - Loads a trained teacher model (with temperature scaling).
    - Initializes a smaller student MLP.
    - Distills teacher knowledge using softened probabilities.
    - Evaluates calibration (ECE, MCS, NLL) and accuracy.
    - Logs results, metrics, and artifacts to MLflow 3.0.

Usage:
    python 03_distillation_experiment.py \
        --teacher_uri runs:/<RUN_ID>/model \
        --train ../../data/train.csv \
        --target label
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow
import mlflow.pytorch

from mugalab.mlp_tuner_tabular_mlflow import MLP, load_tabular_csv, get_device
from mugalab.calibration_metrics import (
    expected_calibration_error,
    miscalibration_score,
    negative_log_likelihood,
)
from mugalab.reliability_diagram_utils import log_reliability_to_mlflow


# ============================================================
# Distillation Loss
# ============================================================

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float
) -> torch.Tensor:
    """Compute combined distillation + hard-label loss.

    Args:
        student_logits: Logits from the student model.
        teacher_logits: Logits from the teacher model.
        labels: True class labels.
        temperature: Temperature for softening logits.
        alpha: Weighting factor between soft and hard targets.

    Returns:
        torch.Tensor: Combined loss.
    """
    soft_loss_fn = nn.KLDivLoss(reduction="batchmean")
    hard_loss_fn = nn.CrossEntropyLoss()

    soft_targets = torch.softmax(teacher_logits / temperature, dim=1)
    log_probs = torch.log_softmax(student_logits / temperature, dim=1)

    soft_loss = soft_loss_fn(log_probs, soft_targets) * (temperature ** 2)
    hard_loss = hard_loss_fn(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ============================================================
# Distillation Training Loop
# ============================================================

def train_student_with_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.7,
    lr: float = 1e-3,
    epochs: int = 50
) -> None:
    """Train student model with knowledge distillation."""
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    student_model.to(device)
    teacher_model.to(device).eval()

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(xb)
            student_logits = student_model(xb)
            loss = distillation_loss(student_logits, teacher_logits, yb, temperature, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Distillation loss: {avg_loss:.4f}")


# ============================================================
# Evaluation Functions
# ============================================================

@torch.no_grad()
def evaluate_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor, task: str) -> dict:
    """Compute accuracy and calibration metrics."""
    device = get_device()
    model.to(device).eval()
    X, y = X.to(device), y.to(device)
    logits = model(X)

    metrics = {
        "accuracy": (logits.argmax(dim=1) == y).float().mean().item(),
        "ECE": expected_calibration_error(logits, y),
        "MCS": miscalibration_score(logits, y),
        "NLL": negative_log_likelihood(logits, y),
    }
    return metrics, logits


# ============================================================
# Experiment Runner
# ============================================================

def run_distillation_experiment(
    teacher_uri: str,
    train_path: str,
    target_col: str,
    mlflow_uri: str = "mlruns",
    experiment_name: str = "Distillation_Experiment",
    temperature: float = 2.0,
    alpha: float = 0.7,
    lr: float = 1e-3,
    epochs: int = 50,
    seed: int = 42
) -> None:
    """Run knowledge distillation and log results to MLflow.

    Args:
        teacher_uri: MLflow URI of the trained teacher model.
        train_path: Path to the training dataset.
        target_col: Target column name.
        mlflow_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        temperature: Distillation temperature.
        alpha: Weight for distillation loss.
        lr: Learning rate for student training.
        epochs: Number of training epochs.
        seed: Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    (X_train, y_train), (X_val, y_val), task = load_tabular_csv(train_path, target_col, seed)
    device = get_device()

    teacher_model = mlflow.pytorch.load_model(teacher_uri)
    teacher_model.to(device).eval()

    input_dim = X_train.shape[1]
    output_dim = int(y_train.max().item() + 1)
    student_model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_units=[64, 64],
        activation="relu",
        dropout=0.2,
        batch_norm=False
    )

    dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    # Train student via knowledge distillation
    train_student_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        dataloader=dataloader,
        device=device,
        temperature=temperature,
        alpha=alpha,
        lr=lr,
        epochs=epochs
    )

    # Evaluate both models
    teacher_metrics, teacher_logits = evaluate_model(teacher_model, X_val, y_val, task)
    student_metrics, student_logits = evaluate_model(student_model, X_val, y_val, task)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="Teacher_Student_Distillation"):
        mlflow.log_params({
            "temperature": temperature,
            "alpha": alpha,
            "learning_rate": lr,
            "epochs": epochs,
            "seed": seed,
        })

        # Log metrics for teacher and student
        prefixed_teacher = {f"teacher_{k}": v for k, v in teacher_metrics.items()}
        prefixed_student = {f"student_{k}": v for k, v in student_metrics.items()}
        mlflow.log_metrics({**prefixed_teacher, **prefixed_student})

        # Log reliability diagram comparison
        log_reliability_to_mlflow(teacher_logits, student_logits, y_val)

        # Save student model
        mlflow.pytorch.log_model(student_model, artifact_path="student_model")
        print("Distillation experiment completed and logged to MLflow.")


# ============================================================
# CLI Entrypoint
# ============================================================

def main():
    """Command-line entry point for distillation experiment."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Experiment")
    parser.add_argument("--teacher_uri", type=str, required=True, help="MLflow teacher model URI or local path.")
    parser.add_argument("--train", type=str, required=True, help="Path to training CSV file.")
    parser.add_argument("--target", type=str, required=True, help="Target column name.")
    parser.add_argument("--mlflow_uri", type=str, default="../../results/mlruns", help="MLflow tracking URI.")
    parser.add_argument("--experiment", type=str, default="Distillation_Experiment", help="MLflow experiment name.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for distillation loss.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for student training.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    run_distillation_experiment(
        teacher_uri=args.teacher_uri,
        train_path=args.train,
        target_col=args.target,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment,
        temperature=args.temperature,
        alpha=args.alpha,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == "__main__":
    main()