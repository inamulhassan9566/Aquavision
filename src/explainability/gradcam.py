"""
AQUAVISION: AI-Based Oil Spill Detection
Grad-CAM (Gradient-Weighted Class Activation Mapping) Module (Phase 8)

Computes visual attention maps to explain convolutional network decisions
without claiming pixel-level segmentation.
"""

from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import cv2


class GradCAM:
    """
    Hook-based Grad-CAM implementation for PyTorch CNN backbones.
    
    Extracts convolutional activation maps and gradients to compute
    class-discriminative localization maps.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple where first element is gradient w.r.t layer output
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generates 2D normalized Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: (1, C, H, W) normalized input tensor.
            target_class: Class ID to explain (if None, uses predicted class).
            
        Returns:
            heatmap (np.ndarray): 2D array in [0.0, 1.0] resized to input (H, W).
            pred_class (int): Predicted class ID.
            confidence (float): Softmax confidence score for pred_class.
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_class].item())

        if target_class is None:
            target_class = pred_class

        # Target score for backpropagation
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients
        # gradients shape: (1, C, H_feat, W_feat)
        # activations shape: (1, C, H_feat, W_feat)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to retain only positive influence
        cam = torch.clamp(cam, min=0.0)
        
        cam_np = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_max = np.max(cam_np)
        if cam_max > 0:
            cam_np = cam_np / cam_max
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to match input spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        return heatmap_resized, pred_class, confidence

    def generate_overlay(
        self,
        original_image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates colorized heatmap and blended overlay on top of original image.
        
        Returns:
            (colored_heatmap_rgb, overlay_rgb)
        """
        if isinstance(original_image, Image.Image):
            orig_np = np.array(original_image)
        else:
            orig_np = original_image.copy()

        # Ensure RGB (3 channels)
        if orig_np.ndim == 2:
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
        elif orig_np.shape[2] == 4:
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)

        # Resize original image to match heatmap dimensions if needed
        h, w = heatmap.shape[:2]
        if orig_np.shape[:2] != (h, w):
            orig_np = cv2.resize(orig_np, (w, h))

        # Convert heatmap to uint8 colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Blend original with colored heatmap
        overlay = (1.0 - alpha) * orig_np.astype(float) + alpha * colored_rgb.astype(float)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return colored_rgb, overlay

    def remove_hooks(self):
        """Cleanly unregisters hooks from model."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
