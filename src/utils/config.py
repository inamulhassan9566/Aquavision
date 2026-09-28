"""
AQUAVISION: AI-Based Oil Spill Detection
Configuration Loader & Validation
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    """Loads configuration from YAML file or raises error if not found."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
