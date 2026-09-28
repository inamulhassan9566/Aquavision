"""
AQUAVISION: AI-Based Oil Spill Detection
Scientific Evaluation Metrics & Curve Generation (Phase 5)

Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix,
and generates publication-quality evaluation figures.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability for positive class (Class 1: Oil Spill).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5

    # Specificity (True Negative Rate)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "specificity": round(specificity, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": cm.tolist()
        },
        "per_class": {
            "no_oil": {
                "precision": round(float(tn / (tn + fn)), 4) if (tn + fn) > 0 else 0.0,
                "recall": round(specificity, 4),
                "support": int(tn + fp)
            },
            "oil_spill": {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "support": int(tp + fn)
            }
        }
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    class_names: List[str] = ["No Oil Spill", "Oil Spill"],
    title: str = "AQUAVISION — Confusion Matrix"
) -> None:
    """Renders and saves a high-contrast confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"}
    )
    
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("True Ground Truth Label", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Model Classification", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    roc_auc: float,
    output_path: Path,
    title: str = "AQUAVISION — Receiver Operating Characteristic (ROC)"
) -> None:
    """Renders and saves ROC curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#0ea5e9", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    title: str = "AQUAVISION — Precision-Recall Curve"
) -> None:
    """Renders and saves Precision-Recall curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(rec, prec, color="#ef4444", lw=2.5, label="Precision-Recall Curve")
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_learning_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "AQUAVISION — Training & Validation Progress"
) -> None:
    """Plots training and validation loss and accuracy across epochs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Plot
    axes[0].plot(epochs, history["train_loss"], "o-", color="#0284c7", lw=2, label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], "s--", color="#dc2626", lw=2, label="Validation Loss")
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True)

    # Accuracy Plot
    axes[1].plot(epochs, history["train_acc"], "o-", color="#0284c7", lw=2, label="Training Accuracy")
    axes[1].plot(epochs, history["val_acc"], "s--", color="#dc2626", lw=2, label="Validation Accuracy")
    axes[1].set_title("Classification Accuracy vs. Epochs", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
