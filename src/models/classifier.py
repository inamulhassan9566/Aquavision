"""
AQUAVISION: AI-Based Oil Spill Detection
Neural Network Classification Architecture (Phase 4)

Implements fine-tuned CNN classifiers with targeted hooks for Grad-CAM explainability.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


class OilSpillClassifier(nn.Module):
    """
    Modular transfer-learning classifier for Sentinel-1 SAR Oil Spill classification.
    
    Supports EfficientNet-B0, ResNet-50, and ConvNeXt-Tiny with calibrated classification
    heads and target convolutional feature hooks for Grad-CAM.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer = self.backbone.features[-1]

        elif self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
            self.target_layer = self.backbone.layer4[-1]

        elif self.architecture == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(in_features, num_classes)
            self.target_layer = self.backbone.features[-1]

        else:
            raise ValueError(f"Unsupported architecture: {architecture}. Choose 'efficientnet_b0', 'resnet50', or 'convnext_tiny'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        return self.backbone(x)

    def get_target_layer_for_gradcam(self) -> nn.Module:
        """Returns the penultimate convolutional layer for Grad-CAM backpropagation."""
        return self.target_layer
