"""
AQUAVISION: AI-Based Oil Spill Detection
Deterministic Seed Management for Complete Reproducibility
"""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and PyTorch (CPU & CUDA)."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
