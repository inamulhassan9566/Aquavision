"""
AQUAVISION: AI-Based Oil Spill Detection
Demo Artifact Generation Pipeline (Phase 8 & 31)

Generates presentation-ready Grad-CAM visual explanations and comparative artifacts.
"""

import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference.predict import OilSpillPredictor


def generate_demo_artifacts():
    output_demo = Path("outputs/demo")
    output_demo.mkdir(parents=True, exist_ok=True)
    test_csv = Path("data/splits/test.csv")
    raw_dir = Path("data/raw")

    df_test = pd.read_csv(test_csv)
    oil_samples = df_test[df_test["class_id"] == 1]
    no_oil_samples = df_test[df_test["class_id"] == 0]

    predictor = OilSpillPredictor.get_instance(Path("models/best_model.pth"))

    # Select representative samples
    test_cases = [
        ("Oil Spill Confirmed Case", oil_samples.iloc[0]),
        ("Clean Ocean / Non-Spill Case", no_oil_samples.iloc[0])
    ]

    for title, row in test_cases:
        img_path = raw_dir / row["relative_path"]
        result = predictor.predict(
            image_input=img_path,
            generate_explanation=True,
            save_artifacts_dir=output_demo
        )

        orig_pil = Image.open(result["image_path"])
        cam_pil = Image.open(result["gradcam_path"])
        overlay_pil = Image.open(result["overlay_path"])

        # Create 3-panel presentation visual
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_pil, cmap="gray")
        axes[0].set_title("1. Original Sentinel-1 SAR Chip", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(cam_pil)
        axes[1].set_title("2. Grad-CAM Attention Heatmap", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay_pil)
        pred_label = result["prediction"]
        conf_pct = result["confidence"] * 100.0
        axes[2].set_title(f"3. Blended Overlay ({pred_label} - {conf_pct:.1f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        safe_name = "oil_spill_demo.png" if row["class_id"] == 1 else "no_oil_demo.png"
        plt.suptitle(f"AQUAVISION AI Explainability — {title}\nDecision: {pred_label} (Confidence: {conf_pct:.2f}%)",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.savefig(output_demo / safe_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated demo artifact: {safe_name}")

    # Copy key evaluation plots to outputs/demo for presentation support
    reports_dir = Path("outputs/reports")
    for plot_name in ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png", "learning_curves.png"]:
        src_file = reports_dir / plot_name
        if src_file.exists():
            shutil.copy(src_file, output_demo / plot_name)

    print("All presentation artifacts generated in outputs/demo/")


if __name__ == "__main__":
    generate_demo_artifacts()
