"""
Exact frozen preprocessing pipeline for DFU Live Segmentation System.
Follows Stage 11/Stage 16 scientific protocols without modification.
"""

from __future__ import annotations

import io
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import torch

from app.core.config import (
    INPUT_WIDTH,
    INPUT_HEIGHT,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class PreprocessingPipeline:
    def __init__(self):
        self.mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)

    def validate_image_bytes(self, image_bytes: bytes) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
        """
        Validates raw image bytes.
        Returns: (is_valid, error_message, pil_image)
        """
        if not image_bytes or len(image_bytes) == 0:
            return False, "Empty image data received", None

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify integrity
            # Reopen after verify() because verify() closes the stream in PIL
            image = Image.open(io.BytesIO(image_bytes))
            if image.width < 16 or image.height < 16:
                return False, f"Image dimensions too small ({image.width}x{image.height})", None
            return True, None, image
        except Exception as e:
            return False, f"Invalid or corrupted image format: {str(e)}", None

    def preprocess_image(self, pil_image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Processes an input PIL image according to the frozen protocol:
        1. Preserves original dimensions (W_orig, H_orig).
        2. Converts to standard 3-channel RGB.
        3. Resizes to frozen resolution (512 x 384) using BILINEAR interpolation.
        4. Scales pixels to [0, 1] float32.
        5. Applies exact ImageNet normalization ((x - mean) / std).
        6. Converts to torch.Tensor in NCHW layout (1, 3, 384, 512).

        Returns: (input_tensor, (orig_width, orig_height))
        """
        orig_width, orig_height = pil_image.size

        # Convert to RGB
        rgb_image = pil_image.convert("RGB")

        # Bilinear resize to 512 x 384
        resized_image = rgb_image.resize(
            (INPUT_WIDTH, INPUT_HEIGHT),
            resample=Image.Resampling.BILINEAR
        )

        # Scale to [0, 1] float32
        arr = np.asarray(resized_image, dtype=np.float32) / 255.0

        # ImageNet normalization
        norm_arr = (arr - self.mean) / self.std

        # Transpose HWC -> CHW and add batch dimension -> NCHW
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(torch.float32)

        return tensor, (orig_width, orig_height)

    def preprocess_ground_truth(self, pil_gt: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes optional ground-truth mask:
        1. Converts to grayscale.
        2. Binarizes at cutoff >= 128 -> 1, else 0.
        Returns:
            - binary_gt_orig: np.ndarray (H_orig, W_orig) uint8 {0, 1}
            - binary_gt_model: np.ndarray (384, 512) uint8 {0, 1} using nearest-neighbor
        """
        gray_gt = np.asarray(pil_gt.convert("L"), dtype=np.uint8)
        binary_gt_orig = np.where(gray_gt >= 128, 1, 0).astype(np.uint8)

        # Model resolution GT via nearest neighbor
        resized_gt_pil = Image.fromarray((binary_gt_orig * 255).astype(np.uint8)).resize(
            (INPUT_WIDTH, INPUT_HEIGHT),
            resample=Image.Resampling.NEAREST
        )
        binary_gt_model = np.where(np.asarray(resized_gt_pil) >= 128, 1, 0).astype(np.uint8)

        return binary_gt_orig, binary_gt_model


# Singleton preprocessor
preprocessing_pipeline = PreprocessingPipeline()
