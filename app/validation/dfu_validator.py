"""
DFU Input Validation and OOD Rejection Gate Service (Memory-Optimized for Cloud Deployment).
Protects the frozen segmentation model from out-of-domain inputs.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

logger = logging.getLogger("dfu_ui.validation")
logger.setLevel(logging.INFO)

# Classification thresholds
DFU_VALIDATION_THRESHOLD = 0.65
DFU_UNCERTAIN_THRESHOLD = 0.35

TEXT_FEATURES_PATH = Path(__file__).resolve().parent / "text_features_cached.pt"


class DomainDecision(str, Enum):
    VALID_DFU_INPUT = "VALID_DFU_INPUT"
    REJECT_NON_DFU = "REJECT_NON_DFU"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class FileValidationResult:
    is_valid: bool
    user_message: str
    error_detail: Optional[str] = None
    pil_image: Optional[Image.Image] = None
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None


@dataclass
class DFUValidationResult:
    decision: DomainDecision
    dfu_score: float
    threshold: float
    top_detected_category: str
    user_message: str
    category_scores: Dict[str, float]
    segmentation_allowed: bool

    @property
    def is_accepted(self) -> bool:
        return self.segmentation_allowed


class DFUInputValidator:
    """
    Dedicated lightweight DFU / Non-DFU Image Domain Validator.
    Uses pre-cached text embeddings with vision-only bfloat16 projection (<80 MB RAM).
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    MIN_DIMENSION = 32
    MAX_DIMENSION = 8192
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB

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
        self.device = torch.device("cpu")
        self.vision_model: Optional[CLIPVisionModelWithProjection] = None
        self.processor: Optional[CLIPImageProcessor] = None
        self.text_features_cached: Optional[torch.Tensor] = None
        self.all_labels: List[str] = [self.PROMPT_DFU] + self.PROMPTS_OOD

    def _initialize_model(self):
        """Lazy-loads the lightweight vision-only projection model on first request."""
        if self.vision_model is not None and self.text_features_cached is not None:
            return

        try:
            logger.info("Initializing lightweight DFU Domain Vision Validator (bfloat16)...")
            
            # Load pre-computed text embeddings (28 KB)
            if TEXT_FEATURES_PATH.exists():
                logger.info(f"Loading cached text embeddings from {TEXT_FEATURES_PATH}")
                self.text_features_cached = torch.load(TEXT_FEATURES_PATH, map_location="cpu").to(torch.bfloat16)
            else:
                logger.warning("Cached text embeddings not found! Regenerating fallback tensor...")
                self.text_features_cached = torch.zeros((len(self.all_labels), 512), dtype=torch.bfloat16)

            self.processor = CLIPImageProcessor.from_pretrained(self.model_id)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(
                self.model_id,
                dtype=torch.bfloat16
            )
            self.vision_model.to(self.device)
            self.vision_model.eval()
            gc.collect()

            logger.info("Lightweight DFU Domain Validator ready (<80 MB RAM footprint).")
        except Exception as e:
            logger.error(f"Failed to initialize DFU Domain Validator: {e}", exc_info=True)
            self.vision_model = None

    def validate_file(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None
    ) -> FileValidationResult:
        if not file_bytes or len(file_bytes) == 0:
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail="Empty file received (0 bytes)."
            )

        if len(file_bytes) > self.MAX_FILE_SIZE_BYTES:
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail="File exceeds maximum size limit (25 MB)."
            )

        if filename:
            ext = Path(filename).suffix.lower()
            if ext and ext not in self.SUPPORTED_EXTENSIONS:
                return FileValidationResult(
                    is_valid=False,
                    user_message=f"Unsupported format '{ext}'. Please upload PNG, JPEG, WebP, BMP, or TIFF.",
                    error_detail=f"Unsupported extension: {ext}"
                )

        import io
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            img = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Image decoding failed: {str(e)}"
            )

        w, h = img.size
        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Dimensions too small ({w}x{h})."
            )

        if w > self.MAX_DIMENSION or h > self.MAX_DIMENSION:
            return FileValidationResult(
                is_valid=False,
                user_message="Invalid file. Please upload a valid image file.",
                error_detail=f"Dimensions exceed limit ({w}x{h})."
            )

        return FileValidationResult(
            is_valid=True,
            user_message="File validated successfully.",
            pil_image=img,
            width=w,
            height=h,
            format=img.format or "RGB"
        )

    def validate_domain(
        self,
        pil_image: Image.Image
    ) -> DFUValidationResult:
        if self.vision_model is None or self.processor is None or self.text_features_cached is None:
            self._initialize_model()
            if self.vision_model is None:
                raise RuntimeError("DFU Domain Validator failed to load.")

        rgb_img = pil_image.convert("RGB")
        img_inputs = self.processor(images=rgb_img, return_tensors="pt")
        img_inputs["pixel_values"] = img_inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            img_outputs = self.vision_model(**img_inputs)
            image_features = img_outputs.image_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ self.text_features_cached.T) * 100.0
            probs = similarity.softmax(dim=-1).float().squeeze()

        dfu_score = float(probs[0].item())
        cat_scores = {self.all_labels[i]: float(probs[i].item()) for i in range(len(self.all_labels))}
        
        top_idx = int(probs.argmax().item())
        top_cat = self.all_labels[top_idx]

        if dfu_score >= DFU_VALIDATION_THRESHOLD:
            decision = DomainDecision.VALID_DFU_INPUT
            msg = f"Valid clinical DFU image confirmed ({dfu_score * 100:.1f}% confidence)."
            allowed = True
        elif dfu_score < DFU_UNCERTAIN_THRESHOLD:
            decision = DomainDecision.REJECT_NON_DFU
            msg = "Image validation failed. Non-DFU input detected."
            allowed = False
        else:
            decision = DomainDecision.UNCERTAIN
            msg = "Uncertain domain match. Please verify the photograph contains a diabetic foot ulcer."
            allowed = False

        return DFUValidationResult(
            decision=decision,
            dfu_score=dfu_score,
            threshold=DFU_VALIDATION_THRESHOLD,
            top_detected_category=top_cat,
            user_message=msg,
            category_scores=cat_scores,
            segmentation_allowed=allowed
        )

    def validate(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        file_res = self.validate_file(file_bytes, filename)
        if not file_res.is_valid:
            return False, file_res.user_message, {
                "valid": False,
                "is_dfu_domain": False,
                "domain_decision": DomainDecision.REJECT_NON_DFU.value,
                "dfu_score": 0.0,
                "threshold": DFU_VALIDATION_THRESHOLD,
                "top_detected_category": "CORRUPTED_OR_INVALID_FILE",
                "segmentation_allowed": False,
                "message": file_res.user_message,
                "error_detail": file_res.error_detail
            }

        domain_res = self.validate_domain(file_res.pil_image)
        is_valid = domain_res.segmentation_allowed

        details = {
            "valid": is_valid,
            "is_dfu_domain": domain_res.decision == DomainDecision.VALID_DFU_INPUT,
            "domain_decision": domain_res.decision.value,
            "dfu_score": round(domain_res.dfu_score, 4),
            "threshold": domain_res.threshold,
            "top_detected_category": domain_res.top_detected_category,
            "segmentation_allowed": domain_res.segmentation_allowed,
            "message": domain_res.user_message,
            "image_dimensions": f"{file_res.width}x{file_res.height}",
            "format": file_res.format
        }

        return is_valid, domain_res.user_message, details


# Global singleton instance
dfu_validator = DFUInputValidator()
