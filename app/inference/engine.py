"""
Inference Engine for DFU Live Segmentation System.
Executes frozen model inference, strict 0.50 thresholding,
and faithful mask-to-original nearest-neighbor restoration.
"""

from __future__ import annotations

import io
import time
from typing import Dict, Any, Tuple, Optional
import base64
import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn.functional as F

from app.core.config import (
    INPUT_WIDTH,
    INPUT_HEIGHT,
    FROZEN_THRESHOLD
)
from app.model_registry.registry import model_registry


class InferenceEngine:
    def __init__(self):
        pass

    def _tensor_to_base64_png(self, pil_image: Image.Image) -> str:
        """Encodes a PIL image as a base64 data URI PNG string."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", optimize=True)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"

    def _create_overlay(
        self,
        rgb_original_pil: Image.Image,
        binary_mask_orig: np.ndarray,
        alpha: float = 0.42
    ) -> Image.Image:
        """
        Creates a crisp clinical-grade overlay:
        - Semi-transparent medical cyan/teal tint on positive ulcer regions.
        - High-contrast boundary contour to delineate the exact ulcer margin.
        """
        orig_np = np.asarray(rgb_original_pil).copy()
        h, w, c = orig_np.shape

        # Color: Medical Teal [0, 200, 215]
        overlay_color = np.array([0, 200, 215], dtype=np.uint8)

        # Region mask boolean
        mask_bool = (binary_mask_orig > 0)

        # Blend base image with cyan tint
        blended = orig_np.copy()
        if np.any(mask_bool):
            blended[mask_bool] = (
                (1.0 - alpha) * orig_np[mask_bool] + alpha * overlay_color
            ).astype(np.uint8)

            # Extract crisp boundary contour
            contours, _ = cv2.findContours(
                binary_mask_orig.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
            # Draw boundary contour in bright cyan [0, 255, 255] with thickness 2
            cv2.drawContours(blended, contours, -1, (0, 255, 255), thickness=2)

        return Image.fromarray(blended)

    def _create_probability_heatmap(
        self,
        prob_model: np.ndarray,
        orig_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Visualization-only probability heatmap:
        Uses bilinear interpolation for smooth visual rendering only.
        Does NOT alter the frozen binary segmentation mask.
        """
        orig_w, orig_h = orig_size
        prob_uint8 = (np.clip(prob_model, 0.0, 1.0) * 255).astype(np.uint8)

        # Resize for smooth visualization
        heatmap_resized = cv2.resize(
            prob_uint8,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR
        )

        # Apply colormap (TURBO / JET)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_TURBO)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_rgb)

    def run_inference(
        self,
        input_tensor: torch.Tensor,
        orig_pil_image: Image.Image,
        pil_ground_truth: Optional[Image.Image] = None,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes frozen live inference:
        1. Selects requested or active model.
        2. Measures pure inference latency.
        3. Computes probability map via sigmoid.
        4. Applies frozen 0.50 threshold at model resolution (384 x 512).
        5. Restores binary mask to original dimensions using NEAREST-NEIGHBOR interpolation.
        6. Generates clinical overlay and optional GT visualization.
        """
        start_e2e = time.perf_counter()

        # Resolve model
        target_model_id = model_id or model_registry.active_model_id
        model = model_registry.load_model(target_model_id)
        model_info = model_registry.get_model_entry(target_model_id)
        device = model_registry.device

        # Move input tensor to device
        tensor_device = input_tensor.to(device)

        # Synchronize for exact CUDA timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_inf = time.perf_counter()

        with torch.no_grad():
            if model_info["architecture"] == "SegFormer-B0":
                raw_out = model(pixel_values=tensor_device).logits
                if raw_out.shape[-2:] != tensor_device.shape[-2:]:
                    raw_out = F.interpolate(
                        raw_out,
                        size=tensor_device.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                logits = raw_out
            else:
                logits = model(tensor_device)

            # Sigmoid activation to probability map
            prob_tensor = torch.sigmoid(logits)

            # Frozen thresholding at model resolution (384 x 512)
            binary_mask_tensor = (prob_tensor >= FROZEN_THRESHOLD).to(torch.uint8)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inf_latency_ms = (time.perf_counter() - start_inf) * 1000.0

        # Extract numpy arrays at model resolution (384, 512)
        prob_model_np = prob_tensor.squeeze().cpu().numpy()
        binary_mask_model_np = binary_mask_tensor.squeeze().cpu().numpy()

        orig_w, orig_h = orig_pil_image.size

        # -------------------------------------------------------------
        # CRITICAL MASK-RESIZING RULE:
        # Resize BINARY MASK to original image dimensions using NEAREST-NEIGHBOR!
        # Do NOT bilinear-resize the final binary mask.
        # -------------------------------------------------------------
        mask_pil_model = Image.fromarray((binary_mask_model_np * 255).astype(np.uint8), mode="L")
        mask_pil_orig = mask_pil_model.resize(
            (orig_w, orig_h),
            resample=Image.Resampling.NEAREST
        )
        binary_mask_orig_np = (np.asarray(mask_pil_orig, dtype=np.uint8) >= 128).astype(np.uint8)

        # Generate Clinical Overlay
        overlay_pil = self._create_overlay(
            orig_pil_image.convert("RGB"),
            binary_mask_orig_np
        )

        # Generate Visualization Probability Heatmap
        heatmap_pil = self._create_probability_heatmap(
            prob_model_np,
            (orig_w, orig_h)
        )

        # Handle Ground Truth if provided
        gt_orig_pil: Optional[Image.Image] = None
        gt_b64: Optional[str] = None
        if pil_ground_truth is not None:
            gray_gt = np.asarray(pil_ground_truth.convert("L"), dtype=np.uint8)
            binary_gt_orig = np.where(gray_gt >= 128, 255, 0).astype(np.uint8)
            gt_orig_pil = Image.fromarray(binary_gt_orig, mode="L")
            gt_b64 = self._tensor_to_base64_png(gt_orig_pil)

        # Encode visualizations to Base64 PNGs for instantaneous frontend rendering
        orig_b64 = self._tensor_to_base64_png(orig_pil_image.convert("RGB"))
        pred_mask_b64 = self._tensor_to_base64_png(mask_pil_orig)
        overlay_b64 = self._tensor_to_base64_png(overlay_pil)
        heatmap_b64 = self._tensor_to_base64_png(heatmap_pil)

        # Calculate metrics
        positive_pixels = int(np.sum(binary_mask_orig_np))
        total_pixels = orig_w * orig_h
        ulcer_area_percentage = round((positive_pixels / max(total_pixels, 1)) * 100.0, 2)

        end_e2e = time.perf_counter()
        e2e_latency_ms = (end_e2e - start_e2e) * 1000.0

        return {
            "model_id": target_model_id,
            "architecture": model_info["architecture"],
            "reader_facing_label": model_info["reader_facing_label"],
            "condition": model_info["condition"],
            "seed": model_info["seed"],
            "input_resolution": f"{INPUT_WIDTH} × {INPUT_HEIGHT}",
            "threshold": FROZEN_THRESHOLD,
            "threshold_status": model_info["threshold_status"],
            "device": "NVIDIA CUDA GPU" if device.type == "cuda" else "Host CPU",
            "export_format": model_info["export_format"],
            "model_sha256": model_info["sha256"],
            "certification_badge": model_info["certification_badge"],
            "original_dimensions": {"width": orig_w, "height": orig_h},
            "mask_dimensions": {"width": orig_w, "height": orig_h},
            "model_resolution_dimensions": {"width": INPUT_WIDTH, "height": INPUT_HEIGHT},
            "raw_logits_model": logits.cpu(),
            "prob_map_model": prob_tensor.cpu(),
            "binary_mask_model": binary_mask_tensor.cpu(),
            "binary_mask_orig_pil": mask_pil_orig,
            "overlay_pil": overlay_pil,
            "original_image_b64": orig_b64,
            "ground_truth_b64": gt_b64,
            "predicted_mask_b64": pred_mask_b64,
            "overlay_b64": overlay_b64,
            "probability_heatmap_b64": heatmap_b64,
            "timing": {
                "inference_latency_ms": round(inf_latency_ms, 2),
                "end_to_end_latency_ms": round(e2e_latency_ms, 2),
                "positive_pixels": positive_pixels,
                "ulcer_area_percentage": ulcer_area_percentage
            }
        }


# Singleton inference engine
inference_engine = InferenceEngine()
