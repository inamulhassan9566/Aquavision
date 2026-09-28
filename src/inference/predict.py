"""
AQUAVISION: AI-Based Oil Spill Detection
Reusable Production-Ready Inference Engine (Phase 9)

Implements cached model loading, fast evaluation, and automated Grad-CAM generation.
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.models.model_factory import build_model
from src.data.preprocessing import get_sar_transforms
from src.explainability.gradcam import GradCAM


class OilSpillPredictor:
    """
    Cached, thread-safe inference engine for AQUAVISION.
    Loads the trained model checkpoint once and reuses it for high-throughput inference.
    """

    _instance: Optional["OilSpillPredictor"] = None

    def __init__(
        self,
        checkpoint_path: Path = Path("models/best_model.pth"),
        device: Optional[str] = None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load Checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.architecture = checkpoint.get("architecture", "efficientnet_b0")
        self.epoch = checkpoint.get("epoch", 1)

        # Build Model
        self.model = build_model(architecture=self.architecture, num_classes=2, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Setup Grad-CAM Hook
        target_layer = self.model.get_target_layer_for_gradcam()
        self.gradcam = GradCAM(self.model, target_layer)

        # Transforms
        self.transform = get_sar_transforms(image_size=224, is_training=False)

    @classmethod
    def get_instance(cls, checkpoint_path: Path = Path("models/best_model.pth")) -> "OilSpillPredictor":
        """Singleton accessor for cached model inference."""
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    def predict(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        generate_explanation: bool = True,
        save_artifacts_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and visual attention generation.
        
        Args:
            image_input: File path, byte stream, or PIL Image.
            generate_explanation: Whether to produce Grad-CAM heatmap & overlay.
            save_artifacts_dir: Directory to save generated images (optional).
            
        Returns:
            Structured prediction dictionary matching the system specification.
        """
        # 1. Parse Image
        if isinstance(image_input, (str, Path)):
            raw_path = Path(image_input)
            pil_img = Image.open(raw_path)
            orig_filename = raw_path.name
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input))
            orig_filename = f"upload_{int(time.time()*1000)}.jpg"
        elif isinstance(image_input, Image.Image):
            pil_img = image_input
            orig_filename = f"image_{int(time.time()*1000)}.jpg"
        else:
            raise ValueError("Unsupported image input type.")

        # Ensure RGB (replicated grayscale)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # 2. Preprocess
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # 3. Predict & Explain
        gradcam_path = None
        overlay_path = None
        original_saved_path = None

        if generate_explanation:
            # Enable gradients for Grad-CAM
            with torch.enable_grad():
                tensor.requires_grad = True
                heatmap, class_id, confidence = self.gradcam.generate_heatmap(tensor)

            probs_all = torch.softmax(self.model(tensor), dim=1).detach().cpu().numpy()[0]
            prob_oil = float(probs_all[1])
            prob_no_oil = float(probs_all[0])
            pred_class_id = class_id

            # Save Visualizations if requested
            if save_artifacts_dir is not None:
                save_dir = Path(save_artifacts_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time() * 1000)

                orig_save_name = f"orig_{timestamp}_{orig_filename}"
                gradcam_save_name = f"gradcam_{timestamp}_{orig_filename}"
                overlay_save_name = f"overlay_{timestamp}_{orig_filename}"

                # Save original
                orig_file = save_dir / orig_save_name
                pil_img.save(orig_file)
                original_saved_path = str(orig_file)

                # Generate and save overlay
                colored_cam, overlay_img = self.gradcam.generate_overlay(pil_img, heatmap)
                overlay_pil = Image.fromarray(overlay_img)
                overlay_file = save_dir / overlay_save_name
                overlay_pil.save(overlay_file)
                overlay_path = str(overlay_file)

                # Save heatmap
                cam_pil = Image.fromarray(colored_cam)
                cam_file = save_dir / gradcam_save_name
                cam_pil.save(cam_file)
                gradcam_path = str(cam_file)
        else:
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_class_id = int(np.argmax(probs))
                confidence = float(probs[pred_class_id])
                prob_oil = float(probs[1])
                prob_no_oil = float(probs[0])

        class_name = "Oil Spill" if pred_class_id == 1 else "No Oil Spill"

        explanation_text = (
            "The model identified visual patterns associated with the Oil Spill class (e.g. characteristic dark backscatter dampening)."
            if pred_class_id == 1 else
            "The model identified visual patterns characteristic of clean open ocean or natural sea surface look-alikes."
        )

        return {
            "prediction": class_name,
            "class_id": pred_class_id,
            "confidence": confidence,
            "probabilities": {
                "oil_spill": prob_oil,
                "no_oil": prob_no_oil
            },
            "explanation": explanation_text,
            "model_version": f"{self.architecture}-v1.0",
            "image_path": original_saved_path,
            "gradcam_path": gradcam_path,
            "overlay_path": overlay_path
        }


def predict(
    image_input: Union[str, Path, bytes, Image.Image],
    checkpoint_path: Path = Path("models/best_model.pth"),
    generate_explanation: bool = True,
    save_artifacts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Helper wrapper function for one-off predictions."""
    predictor = OilSpillPredictor.get_instance(checkpoint_path)
    return predictor.predict(
        image_input=image_input,
        generate_explanation=generate_explanation,
        save_artifacts_dir=save_artifacts_dir
    )
