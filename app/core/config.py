"""
Core configuration for DFU Live Segmentation System Backend.
Strictly isolates application paths and respects frozen research parameters.
"""

from pathlib import Path
from typing import Tuple, List

# Cloud/Local dynamic project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
EXPORTS_DIR = PROJECT_ROOT / "exports"
MODELS_CATALOG_FILE = CONFIG_DIR / "models.json"
DEMO_SAMPLES_FILE = CONFIG_DIR / "demo_samples.json"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen Scientific Parameters (Do NOT modify)
INPUT_WIDTH: int = 512
INPUT_HEIGHT: int = 384
FROZEN_THRESHOLD: float = 0.50
THRESHOLD_STATUS_LABEL: str = "Threshold: 0.50 — Frozen"

# Preprocessing Normalization (ImageNet)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Security and Upload Boundaries
MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB
ALLOWED_MIME_TYPES: List[str] = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff"
]

# Compute Hardware Mode: CPU-Only
USE_CPU_ONLY: bool = True

# API Settings
API_TITLE: str = "DFU Live Segmentation System API"
API_VERSION: str = "1.0.0"
CORS_ORIGINS: List[str] = ["*"]
