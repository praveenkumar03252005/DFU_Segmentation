"""
DFU Input Domain & Image Validation Module
==========================================

Upstream validation gate for Diabetic Foot Ulcer (DFU) Live Segmentation System.
Ensures invalid, out-of-distribution (OOD), corrupted, or non-DFU images
(e.g., cats, dogs, faces, cars, landscapes, screenshots, documents) are strictly
rejected BEFORE reaching the frozen U-Net segmentation model.

CRITICAL ARCHITECTURAL CONSTRAINTS:
-----------------------------------
1. The frozen segmentation model is NEVER modified, retrained, or altered.
2. The validator runs strictly upstream as a safety gate.
3. Segmentation-output heuristics (e.g., mask area, empty mask) are NEVER used as validators.
4. If domain validation fails or is uncertain, segmentation inference is strictly BLOCKED.
"""

from __future__ import annotations

import io
import logging
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger("dfu_ui.validation")
logger.setLevel(logging.INFO)

# ==============================================================================
# 5. CONFIGURABLE VALIDATION THRESHOLDS
# ==============================================================================
# NOTE: These are configurable engineering thresholds selected to cleanly separate
# authentic clinical DFU photographs from out-of-distribution images (animals, faces,
# landscapes, documents, objects). They can be calibrated on specific validation cohorts
# without touching the frozen segmentation model.
DFU_VALIDATION_THRESHOLD: float = 0.65    # Score >= 0.65 -> ACCEPT (VALID_DFU_INPUT)
DFU_UNCERTAIN_THRESHOLD: float = 0.35     # Score < 0.35 -> REJECT (INVALID_NON_DFU_INPUT)
                                         # 0.35 <= Score < 0.65 -> UNCERTAIN_INPUT

ALLOW_UNCERTAIN_INFERENCE: bool = False   # Safety policy: reject uncertain inputs


class DomainDecision(str, Enum):
    ACCEPT = "VALID_DFU_INPUT"
    REJECT = "INVALID_NON_DFU_INPUT"
    UNCERTAIN = "UNCERTAIN_INPUT"


@dataclass
class FileValidationResult:
    is_valid: bool
    user_message: str
    error_detail: Optional[str] = None
    pil_image: Optional[Image.Image] = None
    width: int = 0
    height: int = 0
    format: str = ""


@dataclass
class DFUValidationResult:
    is_accepted: bool
    decision: DomainDecision
    dfu_score: float
    threshold: float
    top_detected_category: str
    user_message: str
    category_scores: Dict[str, float]
    segmentation_allowed: bool


class DFUInputValidator:
    """
    Dedicated lightweight DFU / Non-DFU Image Domain Validator.
    Uses Contrastive Vision-Language representations (CLIP) with cached text embeddings
    for near-instantaneous (10-20ms) zero-shot domain discrimination.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    MIN_DIMENSION = 32
    MAX_DIMENSION = 8192
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

    # Domain Prompts: 1 Target In-Domain category vs comprehensive Out-of-Distribution categories
    PROMPT_DFU = "a clinical medical photograph of a diabetic foot ulcer, foot wound, or foot lesion"
    PROMPTS_OOD = [
        "a photograph of a cat, kitten, or feline animal",
        "a photograph of a dog, puppy, or canine animal",
        "a photograph of a wild animal, bird, reptile, or fish",
        "a portrait photograph of a human face, selfie, or person",
        "a landscape photograph of nature, trees, sky, or mountain scenery",
        "a photograph of a car, truck, vehicle, or automobile",
        "a photograph of food, dish, fruit, or meal",
        "a screenshot of a computer screen, software, website, text document, code, or pdf",
        "a chart, diagram, infographic, presentation slide, or graphic design",
        "a photograph of an indoor room, furniture, or household object",
        "an architectural photograph of a building, city street, or road",
        "an unrelated medical radiograph, x-ray, mri, or ct scan"
    ]

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        # CPU-Only Mode: strictly force torch.device("cpu")
        self.device = torch.device("cpu")
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.text_features_cached: Optional[torch.Tensor] = None
        self.all_labels: List[str] = [self.PROMPT_DFU] + self.PROMPTS_OOD

        # Lazy load model on demand or warm-up
        self._initialize_model()

    def _initialize_model(self):
        """Initializes and caches the vision-language validator."""
        try:
            logger.info(f"Initializing DFU Domain Validator using {self.model_id} on {self.device}...")
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            # Pre-compute and normalize text embeddings for instantaneous image-only inference
            text_inputs = self.processor(
                text=self.all_labels,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                self.text_features_cached = text_features / text_features.norm(dim=-1, keepdim=True)

            logger.info("DFU Domain Validator initialized with cached text embeddings. Upstream safety gate ready.")
        except Exception as e:
            logger.error(f"Failed to initialize DFU Domain Validator: {e}", exc_info=True)
            self.model = None

    # ==========================================================================
    # 1. BASIC FILE INTEGRITY VALIDATION
    # ==========================================================================
    def validate_file(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None
    ) -> FileValidationResult:
        """
        Validates raw file before any AI processing:
        - Non-empty
        - Supported format
        - Decodable without corruption
        - Dimensions in valid range
        """
        if not file_bytes or len(file_bytes) == 0:
            logger.warning("[FILE VALIDATION: FAIL] Uploaded file is empty (0 bytes).")
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail="Empty file received (0 bytes)."
            )

        if len(file_bytes) > self.MAX_FILE_SIZE_BYTES:
            logger.warning(f"[FILE VALIDATION: FAIL] File size exceeds {self.MAX_FILE_SIZE_BYTES} bytes.")
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail="File exceeds maximum size limit (25 MB)."
            )

        # Check file extension if provided
        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning(f"[FILE VALIDATION: FAIL] Unsupported extension {ext}.")
                return FileValidationResult(
                    is_valid=False,
                    user_message="Invalid file. Please upload a valid image file.",
                    error_detail=f"Unsupported image format '{ext}'. Allowed: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
                )

        # Decode and verify integrity with PIL
        try:
            stream = io.BytesIO(file_bytes)
            img = Image.open(stream)
            img.verify()  # Structural integrity check

            # Re-open stream since verify() invalidates the PIL object
            stream.seek(0)
            img = Image.open(stream)
            img.load()   # Force decompression of all pixel data
        except Exception as e:
            logger.warning(f"[FILE VALIDATION: FAIL] Corrupted or undecodable image: {e}")
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Image decoding failed: {str(e)}"
            )

        # Check image dimensions
        w, h = img.size
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            logger.warning(f"[FILE VALIDATION: FAIL] Image dimensions too small ({w}x{h}).")
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Dimensions too small ({w}x{h}). Minimum required: {self.MIN_DIMENSION}x{self.MIN_DIMENSION}."
            )

        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            logger.warning(f"[FILE VALIDATION: FAIL] Image dimensions too large ({w}x{h}).")
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Dimensions exceed limit ({w}x{h})."
            )

        logger.info(f"[FILE VALIDATION: PASS] {w}x{h} {img.format} successfully validated.")
        return FileValidationResult(
            is_valid=True,
            user_message="File validated successfully.",
            pil_image=img,
            width=w,
            height=h,
            format=img.format or "RGB"
        )

    # ==========================================================================
    # 2. DFU DOMAIN VALIDATION (OOD & NON-DFU REJECTION GATE)
    # ==========================================================================
    def validate_domain(
        self,
        pil_image: Image.Image
    ) -> DFUValidationResult:
        """
        Evaluates whether the image falls within the valid clinical DFU input domain.
        Returns: DFUValidationResult containing decision, confidence score, and user messages.
        """
        if self.model is None or self.processor is None or self.text_features_cached is None:
            self._initialize_model()
            if self.model is None:
                raise RuntimeError("DFU Domain Validator failed to load.")

        rgb_img = pil_image.convert("RGB")

        # Encode image features
        img_inputs = self.processor(
            images=rgb_img,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**img_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity scaled by logit_scale
            logit_scale = self.model.logit_scale.exp()
            logits = torch.matmul(image_features, self.text_features_cached.t()) * logit_scale
            probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        dfu_score = float(probabilities[0])
        top_idx = int(np.argmax(probabilities))
        top_label = self.all_labels[top_idx]

        category_scores = {
            label: float(probabilities[i])
            for i, label in enumerate(self.all_labels)
        }

        # ----------------------------------------------------------------------
        # Decision Policy:
        # ----------------------------------------------------------------------
        if dfu_score >= DFU_VALIDATION_THRESHOLD:
            decision = DomainDecision.ACCEPT
            is_accepted = True
            user_message = "Valid diabetic foot ulcer image verified."
            segmentation_allowed = True
        elif dfu_score < DFU_UNCERTAIN_THRESHOLD:
            decision = DomainDecision.REJECT
            is_accepted = False
            user_message = "Invalid image. Please upload a valid diabetic foot ulcer image/file."
            segmentation_allowed = False
        else:
            decision = DomainDecision.UNCERTAIN
            is_accepted = False
            user_message = "Unable to verify this image as a valid diabetic foot ulcer image. Please upload a clear clinical DFU image."
            segmentation_allowed = ALLOW_UNCERTAIN_INFERENCE

        # ----------------------------------------------------------------------
        # 10. STRUCTURED AUDIT LOGGING
        # ----------------------------------------------------------------------
        logger.info(
            f"[DOMAIN VALIDATION] DFU Score: {dfu_score*100:.2f}% | "
            f"Configured Threshold: {DFU_VALIDATION_THRESHOLD*100:.1f}% | "
            f"Top Category: '{top_label}' ({probabilities[top_idx]*100:.2f}%) | "
            f"Decision: {decision.value} | "
            f"Segmentation Executed: {'YES' if segmentation_allowed else 'NO'}"
        )

        return DFUValidationResult(
            is_accepted=is_accepted,
            decision=decision,
            dfu_score=round(dfu_score, 4),
            threshold=DFU_VALIDATION_THRESHOLD,
            top_detected_category=top_label,
            user_message=user_message,
            category_scores=category_scores,
            segmentation_allowed=segmentation_allowed
        )


# Singleton Validator instance
dfu_validator = DFUInputValidator()
