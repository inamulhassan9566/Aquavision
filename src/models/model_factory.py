"""
AQUAVISION: AI-Based Oil Spill Detection
Model Factory Module
"""

import torch.nn as nn
from src.models.classifier import OilSpillClassifier


def build_model(
    architecture: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2
) -> nn.Module:
    """Instantiates and returns the configured model architecture."""
    return OilSpillClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
