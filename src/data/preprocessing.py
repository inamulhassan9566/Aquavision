"""
AQUAVISION: AI-Based Oil Spill Detection
SAR-Aware Image Preprocessing & Augmentation Pipeline (Phase 3)

Preserves 1-band microwave backscatter characteristics while adapting
cleanly to transfer-learning vision backbones.
"""

from typing import Tuple, Callable
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_sar_transforms(
    image_size: int = 224,
    is_training: bool = False
) -> T.Compose:
    """
    Returns image transformation pipeline tailored for Sentinel-1 SAR imagery.
    
    Training augmentations:
    - Random horizontal & vertical flips (radar imaging geometry invariance over ocean)
    - Mild random rotation (+/- 15 degrees)
    - Mild brightness & contrast jitter (simulating radar speckle / calibration variation)
    - NO color hue shift (preserves radiometric backscatter consistency)
    
    Validation/Test:
    - Purely deterministic resize and normalization.
    """
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
