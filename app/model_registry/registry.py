"""
Model Registry Service for DFU Live Segmentation System.
Handles model discovery, SHA-256 integrity checks, and device placement.
Strictly respects Stage16 deployment status and frozen research guardrails.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation

from app.core.config import PROJECT_ROOT, MODELS_CATALOG_FILE

logger = logging.getLogger("dfu_ui.model_registry")
logger.setLevel(logging.INFO)


class ModelRegistry:
    def __init__(self):
        self.catalog: List[Dict[str, Any]] = []
        self.models_by_id: Dict[str, Dict[str, Any]] = {}
        self.loaded_models: Dict[str, torch.nn.Module] = {}
        self.active_model_id: str = "unet_boundary_seed42_pt2"
        # CPU-Only Mode: strictly force torch.device("cpu")
        self.device: torch.device = torch.device("cpu")
        self.integrity_verified: Dict[str, bool] = {}
        self._load_catalog()

    def _compute_sha256(self, file_path: Path) -> str:
        """Computes the SHA-256 hash of a file efficiently in 8MB chunks."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8 * 1024 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _load_catalog(self):
        """Loads and verifies the model catalog from config/models.json."""
        if not MODELS_CATALOG_FILE.exists():
            raise FileNotFoundError(f"Model catalog file not found at {MODELS_CATALOG_FILE}")

        with open(MODELS_CATALOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.catalog = data.get("models", [])
        for entry in self.catalog:
            model_id = entry["model_id"]
            self.models_by_id[model_id] = entry

            # Verify physical existence
            rel_path = entry["model_path"]
            abs_path = PROJECT_ROOT / rel_path

            if not abs_path.exists():
                logger.error(f"Model file missing for {model_id} at {abs_path}")
                self.integrity_verified[model_id] = False
                continue

            # Verify SHA-256 hash
            expected_hash = entry.get("sha256")
            if expected_hash:
                actual_hash = self._compute_sha256(abs_path)
                if actual_hash.lower() == expected_hash.lower():
                    logger.info(f"Integrity verified for {model_id} (SHA-256: {actual_hash[:12]}...)")
                    self.integrity_verified[model_id] = True
                else:
                    logger.error(
                        f"Integrity MISMATCH for {model_id}! Expected {expected_hash}, got {actual_hash}"
                    )
                    self.integrity_verified[model_id] = False
            else:
                self.integrity_verified[model_id] = True

        # Ensure default model is active
        if "unet_boundary_seed42_pt2" in self.models_by_id:
            self.active_model_id = "unet_boundary_seed42_pt2"
        elif self.catalog:
            self.active_model_id = self.catalog[0]["model_id"]

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Returns safe public model descriptors (no absolute server filesystem paths)."""
        safe_models = []
        for entry in self.catalog:
            safe_entry = {k: v for k, v in entry.items() if k not in ["model_path", "checkpoint_path"]}
            safe_entry["is_active"] = (entry["model_id"] == self.active_model_id)
            safe_entry["integrity_verified"] = self.integrity_verified.get(entry["model_id"], False)
            safe_models.append(safe_entry)
        return safe_models

    def get_model_entry(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models_by_id.get(model_id)

    def get_active_model_info(self) -> Dict[str, Any]:
        entry = self.get_model_entry(self.active_model_id)
        if not entry:
            raise RuntimeError(f"Active model {self.active_model_id} not found in catalog")
        safe_entry = {k: v for k, v in entry.items() if k not in ["model_path", "checkpoint_path"]}
        safe_entry["is_active"] = True
        safe_entry["integrity_verified"] = self.integrity_verified.get(self.active_model_id, False)
        return safe_entry

    def set_active_model(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models_by_id:
            raise ValueError(f"Unknown model_id: {model_id}")
        if not self.integrity_verified.get(model_id, False):
            raise RuntimeError(f"Cannot activate model {model_id}: integrity verification failed.")
        self.active_model_id = model_id
        # Preload the model
        self.load_model(model_id)
        return self.get_active_model_info()

    def verify_model_integrity(self, model_id: str) -> bool:
        entry = self.models_by_id.get(model_id)
        if not entry:
            return False
        abs_path = PROJECT_ROOT / entry["model_path"]
        if not abs_path.exists():
            self.integrity_verified[model_id] = False
            return False
        expected_hash = entry.get("sha256")
        if expected_hash:
            actual_hash = self._compute_sha256(abs_path)
            self.integrity_verified[model_id] = (actual_hash.lower() == expected_hash.lower())
        else:
            self.integrity_verified[model_id] = True
        return self.integrity_verified[model_id]

    def load_model(self, model_id: str) -> torch.nn.Module:
        """Loads and caches the model on the target compute device."""
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        entry = self.models_by_id.get(model_id)
        if not entry:
            raise ValueError(f"Model ID {model_id} not recognized")

        if not self.integrity_verified.get(model_id, False):
            if not self.verify_model_integrity(model_id):
                raise RuntimeError(f"Cannot load unverified or missing model: {model_id}")

        abs_path = PROJECT_ROOT / entry["model_path"]
        export_format = entry.get("export_format", "")

        logger.info(f"Loading model {model_id} ({export_format}) from {abs_path.name} onto {self.device}")

        if "PT2" in export_format or abs_path.suffix == ".pt2":
            # PyTorch 2 Static ExportedProgram (already frozen in eval mode; calling .eval() raises NotImplementedError)
            exported_program = torch.export.load(str(abs_path))
            module = exported_program.module().to(self.device)
            self.loaded_models[model_id] = module
            return module

        elif entry["architecture"] == "DeepLabV3+":
            # DeepLabV3+ smp checkpoint
            model = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None,
                upsampling=4
            )
            ckpt = torch.load(str(abs_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.loaded_models[model_id] = model
            return model

        elif entry["architecture"] == "SegFormer-B0":
            # SegFormer-B0 Hugging Face model
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=1,
                ignore_mismatched_sizes=True
            )
            ckpt = torch.load(str(abs_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.loaded_models[model_id] = model
            return model

        elif entry["architecture"] == "U-Net":
            # U-Net smp checkpoint fallback
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation=None
            )
            ckpt = torch.load(str(abs_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.loaded_models[model_id] = model
            return model

        else:
            raise NotImplementedError(f"Unsupported model architecture: {entry['architecture']}")

    def get_active_model(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Returns loaded active model module and its catalog metadata."""
        model = self.load_model(self.active_model_id)
        info = self.models_by_id[self.active_model_id]
        return model, info

    def get_device_status(self) -> Dict[str, Any]:
        """Returns hardware compute device status (CPU-Only Mode enforced)."""
        return {
            "device_name": "Host CPU (CPU-Only Mode)",
            "device_type": "CPU",
            "is_cuda_available": False,
            "gpu_count": 0,
            "memory_allocated_mib": 0.0,
            "memory_reserved_mib": 0.0
        }


# Singleton registry instance
model_registry = ModelRegistry()
