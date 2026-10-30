"""
FastAPI Routes for DFU Live Segmentation System.
Cleanly isolates endpoints, enforces security limits, and suppresses internal server paths.
"""

from __future__ import annotations

import io
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.core.config import (
    PROJECT_ROOT,
    DEMO_SAMPLES_FILE,
    MAX_UPLOAD_SIZE_BYTES,
    FROZEN_THRESHOLD,
    THRESHOLD_STATUS_LABEL
)
from app.model_registry.registry import model_registry
from app.preprocessing.pipeline import preprocessing_pipeline
from app.inference.engine import inference_engine
from app.services.export_service import export_service
from app.validation import (
    dfu_validator,
    DomainDecision,
    DFU_VALIDATION_THRESHOLD,
    DFU_UNCERTAIN_THRESHOLD
)
from app.schemas.models import (
    ModelInfoSchema,
    DeviceStatusSchema,
    ImageValidationResponse,
    InferenceResponseSchema,
    DemoSampleSchema,
    HealthResponseSchema
)

logger = logging.getLogger("dfu_ui.api")
router = APIRouter()


@router.get("/health", response_model=HealthResponseSchema)
async def health_check():
    """System health check and active model state."""
    active_info = model_registry.get_active_model_info()
    device_info = model_registry.get_device_status()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_model_id": active_info["model_id"],
        "active_architecture": active_info["architecture"],
        "device": device_info["device_name"],
        "threshold": FROZEN_THRESHOLD,
        "threshold_status": THRESHOLD_STATUS_LABEL
    }


@router.get("/models", response_model=List[ModelInfoSchema])
async def list_models():
    """List registered candidate models and their engineering metadata."""
    return model_registry.get_available_models()


@router.get("/models/active", response_model=ModelInfoSchema)
async def get_active_model():
    """Retrieve currently active candidate model."""
    return model_registry.get_active_model_info()


@router.post("/models/select")
async def select_model(model_id: str = Form(...)):
    """Switch active engineering candidate."""
    try:
        updated_info = model_registry.set_active_model(model_id)
        return {"status": "success", "active_model": updated_info}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/system/device", response_model=DeviceStatusSchema)
async def get_device():
    """Detect GPU / CPU compute hardware environment."""
    return model_registry.get_device_status()


@router.get("/demo-samples", response_model=List[DemoSampleSchema])
async def list_demo_samples():
    """Provides certified demonstration cases from external FUSeg and internal datasets."""
    if not DEMO_SAMPLES_FILE.exists():
        return []
    with open(DEMO_SAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "source_category": s["source_category"],
            "has_ground_truth": s["has_ground_truth"],
            "description": s["description"]
        }
        for s in samples
    ]


@router.get("/demo-samples/{sample_id}/image")
async def get_demo_image(sample_id: str):
    """Safely retrieves a demonstration image."""
    if not DEMO_SAMPLES_FILE.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Demo samples catalog missing")
    with open(DEMO_SAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = next((s for s in data.get("samples", []) if s["id"] == sample_id), None)
    if not sample:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample not found")

    image_path = PROJECT_ROOT / sample["image_rel_path"]
    if not image_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample image file not found on server")

    media_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    return FileResponse(image_path, media_type=media_type)


@router.get("/demo-samples/{sample_id}/ground-truth")
async def get_demo_ground_truth(sample_id: str):
    """Safely retrieves demonstration ground-truth mask if available."""
    if not DEMO_SAMPLES_FILE.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Demo samples catalog missing")
    with open(DEMO_SAMPLES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample = next((s for s in data.get("samples", []) if s["id"] == sample_id), None)
    if not sample or not sample.get("gt_rel_path"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ground truth available for this sample")

    gt_path = PROJECT_ROOT / sample["gt_rel_path"]
    if not gt_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ground truth mask file not found on server")

    return FileResponse(gt_path, media_type="image/png")


@router.post("/validate-image", response_model=ImageValidationResponse)
async def validate_image(image: UploadFile = File(...)):
    """
    Preflight image validation without executing inference.
    1. Validates file integrity, decodability, dimensions, and size.
    2. Runs upstream DFU domain validator to reject out-of-domain images (cats, dogs, faces, etc.).
    """
    contents = await image.read()

    # Step 1: Basic file integrity validation
    file_val = dfu_validator.validate_file(contents, image.filename)
    if not file_val.is_valid or file_val.pil_image is None:
        logger.warning(f"[API VALIDATE: REJECT FILE] {file_val.error_detail}")
        return {
            "valid": False,
            "is_dfu_domain": False,
            "domain_decision": DomainDecision.REJECT.value,
            "dfu_score": 0.0,
            "threshold": DFU_VALIDATION_THRESHOLD,
            "top_detected_category": "Corrupted or non-image file",
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "format": "INVALID",
            "channels": 0,
            "message": "Invalid file. Please upload a valid image file."
        }

    pil_img = file_val.pil_image
    channels = len(pil_img.getbands())
    aspect_ratio = round(pil_img.width / max(pil_img.height, 1), 3)

    # Step 2: Upstream DFU domain validation
    domain_val = dfu_validator.validate_domain(pil_img)
    logger.info(
        f"[API VALIDATE: DOMAIN] Decision: {domain_val.decision.value} | "
        f"DFU Score: {domain_val.dfu_score*100:.2f}% | "
        f"Allowed: {domain_val.segmentation_allowed}"
    )

    return {
        "valid": domain_val.segmentation_allowed,
        "is_dfu_domain": domain_val.is_accepted,
        "domain_decision": domain_val.decision.value,
        "dfu_score": domain_val.dfu_score,
        "threshold": domain_val.threshold,
        "top_detected_category": domain_val.top_detected_category,
        "width": pil_img.width,
        "height": pil_img.height,
        "aspect_ratio": aspect_ratio,
        "format": pil_img.format or "RGB",
        "channels": channels,
        "message": domain_val.user_message
    }


@router.post("/segment", response_model=InferenceResponseSchema)
async def run_segmentation(
    image: Optional[UploadFile] = File(None),
    ground_truth: Optional[UploadFile] = File(None),
    model_id: Optional[str] = Form(None),
    source_category: str = Form("Demonstration Image"),
    sample_id: Optional[str] = Form(None)
):
    """
    Main live inference endpoint.
    Strictly enforces upstream file and DFU domain validation before running inference.
    The frozen U-Net segmentation model is NEVER executed on invalid or non-DFU inputs.
    """
    pil_img: Optional[Image.Image] = None
    orig_filename: str = "uploaded_case.png"
    pil_gt: Optional[Image.Image] = None

    # Option A: Loaded via demo sample_id
    if sample_id:
        if not DEMO_SAMPLES_FILE.exists():
            raise HTTPException(status_code=404, detail="Demo samples catalog missing")
        with open(DEMO_SAMPLES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        sample = next((s for s in data.get("samples", []) if s["id"] == sample_id), None)
        if not sample:
            raise HTTPException(status_code=404, detail=f"Sample ID {sample_id} not found")

        img_abs = PROJECT_ROOT / sample["image_rel_path"]
        if not img_abs.exists():
            raise HTTPException(status_code=404, detail="Demonstration image file not found")
        pil_img = Image.open(img_abs)
        orig_filename = img_abs.name
        source_category = sample["source_category"]

        # Domain verification for demo sample
        domain_val = dfu_validator.validate_domain(pil_img)
        if not domain_val.segmentation_allowed:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=domain_val.user_message
            )

        if sample.get("gt_rel_path"):
            gt_abs = PROJECT_ROOT / sample["gt_rel_path"]
            if gt_abs.exists():
                pil_gt = Image.open(gt_abs)

    # Option B: Uploaded directly by researcher
    elif image is not None:
        contents = await image.read()

        # Step 1: Upstream Basic File Validation
        file_val = dfu_validator.validate_file(contents, image.filename)
        if not file_val.is_valid or file_val.pil_image is None:
            logger.warning(f"[SEGMENT GUARD: REJECT FILE] {file_val.error_detail}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file. Please upload a valid image file."
            )

        pil_img = file_val.pil_image
        orig_filename = image.filename or "uploaded_image.png"

        # Step 2: Upstream DFU Domain Validation Gate
        domain_val = dfu_validator.validate_domain(pil_img)
        if not domain_val.segmentation_allowed:
            logger.warning(
                f"[SEGMENT GUARD: REJECT DOMAIN] Decision: {domain_val.decision.value} | "
                f"DFU Score: {domain_val.dfu_score*100:.2f}% < {domain_val.threshold*100:.1f}% | "
                f"Top Category: {domain_val.top_detected_category} | "
                f"Halting inference pipeline before segmentation model."
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=domain_val.user_message
            )

        # Check optional uploaded ground truth
        if ground_truth is not None:
            gt_bytes = await ground_truth.read()
            if len(gt_bytes) > 0:
                is_gt_valid, _, loaded_gt = preprocessing_pipeline.validate_image_bytes(gt_bytes)
                if is_gt_valid and loaded_gt is not None:
                    pil_gt = loaded_gt

    else:
        raise HTTPException(
            status_code=400,
            detail="Either an image upload or a sample_id must be provided."
        )

    # EXPLICIT INFERENCE GUARD:
    # Double-check that domain_val passed and segmentation is strictly allowed
    if 'domain_val' not in locals() or not domain_val.segmentation_allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Validation failed. Segmentation is not allowed for this input."
        )

    # Verify model choice
    target_model_id = model_id or model_registry.active_model_id
    if target_model_id not in model_registry.models_by_id:
        raise HTTPException(status_code=400, detail=f"Unrecognized model ID: {target_model_id}")

    # Check model integrity before executing
    if not model_registry.integrity_verified.get(target_model_id, False):
        raise HTTPException(
            status_code=500,
            detail=f"Model integrity verification failed for {target_model_id}. Execution halted."
        )

    # 1. Preprocess image
    tensor, orig_dims = preprocessing_pipeline.preprocess_image(pil_img)

    # 2. Run frozen inference
    result = inference_engine.run_inference(
        input_tensor=tensor,
        orig_pil_image=pil_img,
        pil_ground_truth=pil_gt,
        model_id=target_model_id
    )

    # 3. Package export in isolated UI directory
    packaged = export_service.package_export(
        inference_result=result,
        orig_pil=pil_img,
        image_name=orig_filename,
        gt_pil=pil_gt
    )

    export_id = packaged["export_id"]

    # Build provenance record
    provenance = {
        "image_identifier": orig_filename,
        "filename": orig_filename,
        "source_category": source_category,
        "model_architecture": result["architecture"],
        "condition": result["condition"],
        "seed": result["seed"],
        "threshold": result["threshold"],
        "input_resolution": result["input_resolution"],
        "model_sha256": result["model_sha256"],
        "device": result["device"],
        "export_type": result["export_format"],
        "timestamp": packaged["timestamp"],
        "inference_latency_ms": result["timing"]["inference_latency_ms"],
        "end_to_end_latency_ms": result["timing"]["end_to_end_latency_ms"],
        "certification_status": result["certification_badge"]
    }

    return {
        "image_id": f"img_{uuid.uuid4().hex[:8]}",
        "original_filename": orig_filename,
        "source_category": source_category,
        "model_id": result["model_id"],
        "architecture": result["architecture"],
        "reader_facing_label": result["reader_facing_label"],
        "condition": result["condition"],
        "seed": result["seed"],
        "input_resolution": result["input_resolution"],
        "threshold": result["threshold"],
        "threshold_status": result["threshold_status"],
        "device": result["device"],
        "export_format": result["export_format"],
        "model_sha256": result["model_sha256"],
        "certification_badge": result["certification_badge"],
        "original_dimensions": result["original_dimensions"],
        "mask_dimensions": result["mask_dimensions"],
        "original_image_b64": result["original_image_b64"],
        "ground_truth_b64": result["ground_truth_b64"],
        "predicted_mask_b64": result["predicted_mask_b64"],
        "overlay_b64": result["overlay_b64"],
        "probability_heatmap_b64": result["probability_heatmap_b64"],
        "timing": result["timing"],
        "export_id": export_id,
        "provenance_record": provenance
    }


@router.get("/exports/download/{export_id}/{artifact_type}")
async def download_export_artifact(export_id: str, artifact_type: str):
    """Safely streams user-requested export files."""
    file_path = export_service.get_export_file(export_id, artifact_type)
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail="Requested export artifact not found or invalid export_id"
        )

    media_type = "application/octet-stream"
    if file_path.suffix == ".png":
        media_type = "image/png"
    elif file_path.suffix == ".json":
        media_type = "application/json"
    elif file_path.suffix == ".zip":
        media_type = "application/zip"

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=file_path.name
    )
