"""
Export Service for DFU Live Segmentation System.
Saves user-requested downloads strictly under D:\\DFU_Work\\ui\\exports\\.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import EXPORTS_DIR


class ExportService:
    def __init__(self):
        self.exports_dir = EXPORTS_DIR
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def _create_multi_panel_figure(
        self,
        orig_pil: Image.Image,
        mask_pil: Image.Image,
        overlay_pil: Image.Image,
        gt_pil: Optional[Image.Image] = None,
        metadata_summary: Optional[str] = None
    ) -> Image.Image:
        """
        Creates a clean publication-style multi-panel clinical figure.
        Panels: [Original] | [Ground Truth (if present)] | [Predicted Mask] | [Overlay]
        """
        panels = [("Original Ulcer Image", orig_pil.convert("RGB"))]
        if gt_pil is not None:
            panels.append(("Ground Truth Mask", gt_pil.convert("RGB")))
        panels.append(("Predicted Mask (0.50 Frozen)", mask_pil.convert("RGB")))
        panels.append(("Segmentation Overlay", overlay_pil.convert("RGB")))

        target_h = 384
        resized_panels = []
        for title, p in panels:
            w, h = p.size
            aspect = w / h
            new_w = int(target_h * aspect)
            resized_p = p.resize((new_w, target_h), resample=Image.Resampling.BILINEAR)
            resized_panels.append((title, resized_p))

        spacing = 20
        header_h = 60
        footer_h = 40 if metadata_summary else 20
        total_w = sum(p[1].width for p in resized_panels) + spacing * (len(resized_panels) + 1)
        total_h = target_h + header_h + footer_h

        # Canvas with clean white background
        canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        current_x = spacing
        for title, p in resized_panels:
            # Draw title
            draw.text((current_x + 5, 20), title, fill=(30, 41, 59))
            # Paste panel
            canvas.paste(p, (current_x, header_h))
            # Border outline around each panel
            draw.rectangle(
                [current_x, header_h, current_x + p.width, header_h + target_h],
                outline=(203, 213, 225),
                width=1
            )
            current_x += p.width + spacing

        if metadata_summary:
            draw.text((spacing, total_h - 30), metadata_summary, fill=(100, 116, 139))

        return canvas

    def package_export(
        self,
        inference_result: Dict[str, Any],
        orig_pil: Image.Image,
        image_name: str = "dfu_sample",
        gt_pil: Optional[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Packages prediction artifacts into a dedicated isolated export directory.
        """
        export_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_dir = self.exports_dir / export_id
        target_dir.mkdir(parents=True, exist_ok=True)

        mask_pil = inference_result["binary_mask_orig_pil"]
        overlay_pil = inference_result["overlay_pil"]

        # 1. Save Predicted Mask (PNG)
        mask_path = target_dir / "predicted_mask.png"
        mask_pil.save(mask_path, format="PNG")

        # 2. Save Overlay (PNG)
        overlay_path = target_dir / "segmentation_overlay.png"
        overlay_pil.save(overlay_path, format="PNG")

        # 3. Save Multi-panel Visualization (PNG)
        meta_label = (
            f"DFU Live System | Model: {inference_result['architecture']} ({inference_result['condition']}) | "
            f"Threshold: {inference_result['threshold']} (Frozen) | Device: {inference_result['device']} | "
            f"Latency: {inference_result['timing']['inference_latency_ms']} ms"
        )
        vis_pil = self._create_multi_panel_figure(
            orig_pil=orig_pil,
            mask_pil=mask_pil,
            overlay_pil=overlay_pil,
            gt_pil=gt_pil,
            metadata_summary=meta_label
        )
        vis_path = target_dir / "result_visualization.png"
        vis_pil.save(vis_path, format="PNG")

        # 4. Save Optional Ground Truth if provided
        gt_path: Optional[Path] = None
        if gt_pil is not None:
            gt_path = target_dir / "ground_truth.png"
            gt_pil.save(gt_path, format="PNG")

        # 5. Provenance Metadata Record (JSON)
        timestamp_iso = datetime.now(timezone.utc).isoformat()
        metadata_record = {
            "export_id": export_id,
            "timestamp_utc": timestamp_iso,
            "image_identifier": image_name,
            "original_filename": image_name,
            "architecture": inference_result["architecture"],
            "condition": inference_result["condition"],
            "reader_facing_label": inference_result["reader_facing_label"],
            "seed": inference_result["seed"],
            "threshold": inference_result["threshold"],
            "threshold_status": inference_result["threshold_status"],
            "input_resolution": inference_result["input_resolution"],
            "model_sha256": inference_result["model_sha256"],
            "export_format": inference_result["export_format"],
            "certification_badge": inference_result["certification_badge"],
            "device": inference_result["device"],
            "original_dimensions": inference_result["original_dimensions"],
            "mask_dimensions": inference_result["mask_dimensions"],
            "timing_metrics": inference_result["timing"],
            "scientific_guardrail": "Engineering demonstration record only; does not alter frozen scientific benchmarks."
        }
        meta_path = target_dir / "inference_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_record, f, indent=2)

        # 6. Create convenient ZIP archive containing all artifacts
        zip_path = target_dir / f"dfu_segmentation_{export_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(mask_path, arcname="predicted_mask.png")
            zf.write(overlay_path, arcname="segmentation_overlay.png")
            zf.write(vis_path, arcname="result_visualization.png")
            zf.write(meta_path, arcname="inference_metadata.json")
            if gt_path and gt_path.exists():
                zf.write(gt_path, arcname="ground_truth.png")

        return {
            "export_id": export_id,
            "timestamp": timestamp_iso,
            "metadata_record": metadata_record,
            "files": {
                "mask": f"/api/exports/download/{export_id}/mask",
                "overlay": f"/api/exports/download/{export_id}/overlay",
                "visualization": f"/api/exports/download/{export_id}/visualization",
                "metadata": f"/api/exports/download/{export_id}/metadata",
                "zip": f"/api/exports/download/{export_id}/zip"
            }
        }

    def get_export_file(self, export_id: str, file_type: str) -> Optional[Path]:
        """Safely retrieves an export file avoiding directory traversal."""
        safe_export_id = Path(export_id).name
        target_dir = self.exports_dir / safe_export_id
        if not target_dir.exists() or not target_dir.is_dir():
            return None

        mapping = {
            "mask": "predicted_mask.png",
            "overlay": "segmentation_overlay.png",
            "visualization": "result_visualization.png",
            "metadata": "inference_metadata.json",
            "zip": f"dfu_segmentation_{safe_export_id}.zip"
        }
        filename = mapping.get(file_type)
        if not filename:
            return None

        file_path = target_dir / filename
        return file_path if file_path.exists() else None


# Singleton export service
export_service = ExportService()
