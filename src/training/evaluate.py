"""
AQUAVISION: AI-Based Oil Spill Detection
Independent Test Set Evaluation Pipeline (Phase 5)

Performs single-pass evaluation on the held-out test set (15% split),
computing comprehensive scientific metrics, confusion matrix, and ROC/PR curves.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import seed_everything
from src.utils.logging import setup_logger
from src.utils.config import load_config
from src.data.dataset import Sentinel1SARDataset
from src.data.preprocessing import get_sar_transforms
from src.models.model_factory import build_model
from src.training.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve
)

logger = setup_logger("aquavision.evaluate", log_file=Path("outputs/reports/evaluation.log"))


def evaluate_test_set(
    checkpoint_path: Path = Path("models/best_model.pth"),
    config_path: Path = Path("config.yaml")
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the held-out test dataset exactly once.
    Generates all required report artifacts and curves.
    """
    cfg = load_config(config_path)
    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluation running on device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch = checkpoint.get("architecture", cfg["model"].get("architecture", "efficientnet_b0"))

    model = build_model(architecture=arch, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded best checkpoint from Epoch {checkpoint.get('epoch', 'N/A')} ({arch})")

    # Load Test Set
    splits_dir = Path(cfg["dataset"]["splits_dir"])
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    img_size = cfg["dataset"].get("image_size", 224)
    batch_size = cfg["dataset"].get("batch_size", 32)
    num_workers = cfg["dataset"].get("num_workers", 0)

    test_transform = get_sar_transforms(image_size=img_size, is_training=False)
    test_dataset = Sentinel1SARDataset(splits_dir / "test.csv", raw_dir=raw_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    logger.info(f"Test dataset loaded: {len(test_dataset)} unseen SAR chips")

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for images, labels, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true_all.extend(labels.cpu().numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())
            y_prob_all.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_prob = np.array(y_prob_all)

    # Compute comprehensive metrics
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    logger.info("=== HELD-OUT TEST EVALUATION RESULTS ===")
    logger.info(f"Accuracy:    {metrics['accuracy']*100:.2f}%")
    logger.info(f"Precision:   {metrics['precision']*100:.2f}%")
    logger.info(f"Recall:      {metrics['recall']*100:.2f}%")
    logger.info(f"F1-Score:    {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    logger.info(f"Specificity: {metrics['specificity']*100:.2f}%")
    logger.info(f"Confusion Matrix: {metrics['confusion_matrix']['matrix']}")

    # Save Output Plots
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(metrics["confusion_matrix"]["matrix"])
    plot_confusion_matrix(cm_array, reports_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, metrics["roc_auc"], reports_dir / "roc_curve.png")
    plot_pr_curve(y_true, y_prob, reports_dir / "precision_recall_curve.png")

    # Save test metrics JSON
    with open(reports_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Compile comprehensive model_report.json (Section 31 requirement)
    with open(splits_dir / "split_summary.json", "r", encoding="utf-8") as f:
        split_summary = json.load(f)

    model_report = {
        "project": "AQUAVISION",
        "dataset_name": "Sentinel-1 SAR Oil Spill Detection Dataset",
        "task": "Binary Classification (Oil Spill vs. No Oil Spill)",
        "model_architecture": arch,
        "model_checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch", 1),
        "training_configuration": cfg["training"],
        "dataset_statistics": {
            "total_curated_chips": split_summary["total_samples"],
            "training_samples": split_summary["splits"]["train"]["count"],
            "validation_samples": split_summary["splits"]["validation"]["count"],
            "test_samples": split_summary["splits"]["test"]["count"],
            "oil_spill_percentage": split_summary["splits"]["test"]["class_1_pct"]
        },
        "validation_metrics": checkpoint.get("val_metrics", {}),
        "test_metrics": metrics,
        "limitations": [
            "Trained and evaluated on curated Sentinel-1 SAR chips (400x400) without raw satellite scene geolocation.",
            "Radar look-alikes (low wind zones, biogenic films) can occasionally trigger false positives.",
            "Grad-CAM provides visual attention attribution, not fine-grained pixel-level segmentation."
        ]
    }

    with open(reports_dir / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(model_report, f, indent=2)

    logger.info(f"All reports and visual plots saved to {reports_dir}")
    return model_report


if __name__ == "__main__":
    evaluate_test_set()
