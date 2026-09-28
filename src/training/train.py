"""
AQUAVISION: AI-Based Oil Spill Detection
Model Training & Checkpointing Pipeline (Phase 4)

Implements class-weighted Cross-Entropy loss, AdamW optimization,
Cosine Annealing scheduler, early stopping, and metric tracking.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

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
from src.training.metrics import compute_classification_metrics, plot_learning_curves

logger = setup_logger("aquavision.train", log_file=Path("outputs/reports/training.log"))


def train_model(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Runs end-to-end model training, validation, and checkpointing."""
    cfg = load_config(config_path)
    
    seed = cfg["training"].get("seed", 42)
    seed_everything(seed)

    # Device configuration
    device_req = cfg["training"].get("device", "auto")
    if device_req == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    logger.info(f"Target execution device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Paths
    splits_dir = Path(cfg["dataset"]["splits_dir"])
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = Path(cfg["training"]["outputs_dir"])
    reports_dir = outputs_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = outputs_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Datasets & Loaders
    img_size = cfg["dataset"].get("image_size", 224)
    batch_size = cfg["dataset"].get("batch_size", 32)
    num_workers = cfg["dataset"].get("num_workers", 0)

    train_transform = get_sar_transforms(image_size=img_size, is_training=True)
    val_transform = get_sar_transforms(image_size=img_size, is_training=False)

    train_dataset = Sentinel1SARDataset(splits_dir / "train.csv", raw_dir=raw_dir, transform=train_transform)
    val_dataset = Sentinel1SARDataset(splits_dir / "val.csv", raw_dir=raw_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    logger.info(f"Dataset partitions loaded: {len(train_dataset)} Train samples, {len(val_dataset)} Val samples")

    # Class Imbalance Weights
    use_weights = cfg["training"].get("use_class_weights", True)
    if use_weights:
        class_counts = train_dataset.df["class_id"].value_counts().to_dict()
        n_c0 = class_counts.get(0, 1)
        n_c1 = class_counts.get(1, 1)
        total_n = len(train_dataset)
        w0 = total_n / (2.0 * n_c0)
        w1 = total_n / (2.0 * n_c1)
        weights_tensor = torch.tensor([w0, w1], dtype=torch.float32).to(device)
        logger.info(f"Class-weighted Loss enabled: Class 0 weight={w0:.3f}, Class 1 weight={w1:.3f}")
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # Model instantiation
    arch = cfg["model"].get("architecture", "efficientnet_b0")
    pretrained = cfg["model"].get("pretrained", True)
    dropout = cfg["model"].get("dropout", 0.2)
    model = build_model(architecture=arch, num_classes=2, pretrained=pretrained, dropout=dropout).to(device)
    logger.info(f"Model architecture '{arch}' instantiated (Pretrained={pretrained}, Dropout={dropout})")

    # Freeze early layers for efficient transfer learning if requested
    if cfg["training"].get("freeze_early_layers", False):
        if hasattr(model.backbone, "features"):
            for param in model.backbone.features[:-2].parameters():
                param.requires_grad = False
            logger.info("Transfer learning: Froze early backbone feature stages [:-2], fine-tuning top blocks and classifier head.")

    # Optimizer & Scheduler
    lr = float(cfg["training"].get("learning_rate", 3e-4))
    wd = float(cfg["training"].get("weight_decay", 1e-4))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    epochs = int(cfg["training"].get("epochs", 8))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    early_stopping_patience = cfg["training"].get("early_stopping_patience", 3)

    # Tracking History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_roc_auc": [],
        "lr": []
    }

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    logger.info(f"Commencing training across {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_total = 0.0
        train_correct = 0
        total_train_samples = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train_samples += images.size(0)

        epoch_train_loss = train_loss_total / total_train_samples
        epoch_train_acc = train_correct / total_train_samples

        # 2. Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_labels_all = []
        val_preds_all = []
        val_probs_all = []
        total_val_samples = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_labels_all.extend(labels.cpu().numpy().tolist())
                val_preds_all.extend(preds.cpu().numpy().tolist())
                val_probs_all.extend(probs[:, 1].cpu().numpy().tolist())
                total_val_samples += images.size(0)

        epoch_val_loss = val_loss_total / total_val_samples
        val_metrics = compute_classification_metrics(
            np.array(val_labels_all),
            np.array(val_preds_all),
            np.array(val_probs_all)
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Record history
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(val_metrics["accuracy"], 4))
        history["val_f1"].append(round(val_metrics["f1_score"], 4))
        history["val_precision"].append(round(val_metrics["precision"], 4))
        history["val_recall"].append(round(val_metrics["recall"], 4))
        history["val_roc_auc"].append(round(val_metrics["roc_auc"], 4))
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_metrics['accuracy']*100:.2f}% | "
            f"Val F1: {val_metrics['f1_score']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Checkpointing Best Model (using validation F1)
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "architecture": arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pth")
            logger.info(f"--> Saved best model checkpoint at Epoch {epoch} with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at Epoch {epoch} (No improvement for {early_stopping_patience} epochs)")
                break

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s. Best Epoch: {best_epoch} (Val F1: {best_val_f1:.4f})")

    # Save last model checkpoint as well
    torch.save({
        "epoch": epoch,
        "architecture": arch,
        "model_state_dict": model.state_dict(),
        "config": cfg
    }, checkpoint_dir / "last_model.pth")

    # Save history and learning curves
    with open(reports_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_learning_curves(
        history,
        reports_dir / "learning_curves.png",
        title=f"AQUAVISION — Training Progress ({arch.upper()})"
    )

    # Log Experiment Run
    run_record = {
        "timestamp": datetime.now().isoformat(),
        "architecture": arch,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "training_time_seconds": round(total_time, 2),
        "device": str(device),
        "config": cfg,
        "final_epoch_metrics": {
            "val_accuracy": history["val_acc"][-1],
            "val_f1": history["val_f1"][-1],
            "val_precision": history["val_precision"][-1],
            "val_recall": history["val_recall"][-1],
            "val_roc_auc": history["val_roc_auc"][-1]
        }
    }
    exp_file = exp_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{arch}.json"
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(run_record, f, indent=2)

    logger.info(f"Experiment log saved to {exp_file}")
    return run_record


if __name__ == "__main__":
    train_model()
