"""
AQUAVISION: AI-Based Oil Spill Detection
PyTorch Dataset Definition for Sentinel-1 SAR Chips (Phase 3)
"""

from pathlib import Path
from typing import Optional, Tuple, Callable

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class Sentinel1SARDataset(Dataset):
    """
    Dataset wrapper for Sentinel-1 SAR chips from split CSVs.
    
    Reads 400x400 SAR JPEG chips, guarantees 3-channel input representation
    for deep learning backbones, and returns (tensor, label_int, filename).
    """

    def __init__(
        self,
        csv_file: Path,
        raw_dir: Path = Path("data/raw"),
        transform: Optional[Callable] = None
    ):
        self.csv_file = Path(csv_file)
        self.raw_dir = Path(raw_dir)
        self.transform = transform

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Split CSV file not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        image_path = self.raw_dir / row["relative_path"]
        
        # Load image safely
        with Image.open(image_path) as img:
            # Ensure 3-channel RGB (replicated grayscale)
            if img.mode != "RGB":
                img = img.convert("RGB")
            else:
                img = img.copy()

        label = int(row["class_id"])
        filename = str(row["filename"])

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            import torchvision.transforms.functional as TF
            tensor = TF.to_tensor(img)

        return tensor, label, filename
