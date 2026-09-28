"""
AQUAVISION: AI-Based Oil Spill Detection
Data Splitting & Leakage Prevention Pipeline (Phase 3)

Implements stratified 70/15/15 train/val/test split with deterministic seeding,
leakage verification, and anomaly filtering.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

from src.utils.seed import seed_everything
from src.utils.logging import setup_logger

logger = setup_logger("aquavision.split")


def create_stratified_splits(
    raw_dir: Path = Path("data/raw"),
    splits_dir: Path = Path("data/splits"),
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    filter_zero_variance: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Creates reproducible, stratified train/validation/test splits.
    
    Validates:
    - Zero data leakage (disjoint sets)
    - Stratified class balance preservation
    - Anomaly / zero-variance chip handling
    """
    seed_everything(random_seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    classes = [
        ("Class_0", 0, "No Oil Spill"),
        ("Class_1", 1, "Oil Spill")
    ]

    records: List[Dict] = []
    zero_variance_excluded: List[str] = []

    for folder_name, class_id, class_name in classes:
        cls_path = raw_dir / folder_name
        if not cls_path.exists():
            raise FileNotFoundError(f"Class folder not found: {cls_path}")

        files = sorted(list(cls_path.glob("*.jpg")))
        logger.info(f"Scanning {folder_name}: {len(files)} files found.")

        for f in files:
            # Check for zero-variance artifact
            if filter_zero_variance:
                with Image.open(f) as img:
                    arr = np.array(img)[:, :, 0]
                    if np.std(arr) == 0.0:
                        zero_variance_excluded.append(f.name)
                        continue

            rel_path = f"{folder_name}/{f.name}"
            records.append({
                "filename": f.name,
                "relative_path": rel_path,
                "class_id": class_id,
                "class_name": class_name
            })

    df = pd.DataFrame(records)
    logger.info(f"Total valid samples after filtering: {len(df)} (Excluded {len(zero_variance_excluded)} zero-variance artifacts)")

    # Stratified Split: First split off test set (15%)
    val_plus_test_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=val_plus_test_ratio,
        stratify=df["class_id"],
        random_state=random_seed
    )

    # Next split remaining 30% equally into val (15%) and test (15%)
    test_rel_ratio = test_ratio / val_plus_test_ratio # 0.50
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_rel_ratio,
        stratify=temp_df["class_id"],
        random_state=random_seed
    )

    # Zero Data Leakage Verification
    train_set = set(train_df["filename"])
    val_set = set(val_df["filename"])
    test_set = set(test_df["filename"])

    leakage_train_val = train_set.intersection(val_set)
    leakage_val_test = val_set.intersection(test_set)
    leakage_train_test = train_set.intersection(test_set)

    assert len(leakage_train_val) == 0, f"DATA LEAKAGE DETECTED between Train and Val: {leakage_train_val}"
    assert len(leakage_val_test) == 0, f"DATA LEAKAGE DETECTED between Val and Test: {leakage_val_test}"
    assert len(leakage_train_test) == 0, f"DATA LEAKAGE DETECTED between Train and Test: {leakage_train_test}"
    logger.info("VERIFIED: Zero data leakage. All split partitions are strictly disjoint.")

    # Save to CSV files
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "random_seed": random_seed,
        "total_samples": len(df),
        "zero_variance_excluded_count": len(zero_variance_excluded),
        "splits": {
            "train": {
                "count": len(train_df),
                "ratio": round(len(train_df) / len(df), 4),
                "class_0_count": int((train_df["class_id"] == 0).sum()),
                "class_1_count": int((train_df["class_id"] == 1).sum()),
                "class_1_pct": round(float((train_df["class_id"] == 1).mean() * 100), 2)
            },
            "validation": {
                "count": len(val_df),
                "ratio": round(len(val_df) / len(df), 4),
                "class_0_count": int((val_df["class_id"] == 0).sum()),
                "class_1_count": int((val_df["class_id"] == 1).sum()),
                "class_1_pct": round(float((val_df["class_id"] == 1).mean() * 100), 2)
            },
            "test": {
                "count": len(test_df),
                "ratio": round(len(test_df) / len(df), 4),
                "class_0_count": int((test_df["class_id"] == 0).sum()),
                "class_1_count": int((test_df["class_id"] == 1).sum()),
                "class_1_pct": round(float((test_df["class_id"] == 1).mean() * 100), 2)
            }
        }
    }

    summary_path = splits_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")
    logger.info(f"Split artifacts saved to {splits_dir}")

    return {"train": train_df, "val": val_df, "test": test_df}


if __name__ == "__main__":
    create_stratified_splits()
