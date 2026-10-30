"""
Pydantic schemas for DFU Live Segmentation System.
Ensures strictly typed API inputs and outputs without leaking server file paths.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class InputSizeSchema(BaseModel):
    width: int
    height: int

class EngineeringMetricsSchema(BaseModel):
    parameters: Optional[str] = None
    model_file_size: Optional[str] = None
    checkpoint_size_mib: Optional[str] = None
    peak_gpu_memory_mib: Optional[str] = None
    latency_mean_ms: Optional[str] = None
    latency_median_ms: Optional[str] = None
    throughput_fps: Optional[str] = None
    parity_max_abs_logit_diff: Optional[str] = None
    parity_max_abs_prob_diff: Optional[str] = None
    parity_mismatch_pixels: Optional[int] = None
    parity_verification_cohort: Optional[str] = None

class ModelInfoSchema(BaseModel):
    model_id: str
    architecture: str
    display_name: str
    reader_facing_label: str
    condition: str
    seed: int
    input_size: InputSizeSchema
    threshold: float
    threshold_status: str
    export_format: str
    sha256: str
    checkpoint_sha256: str
    certification_status: str
    certification_badge: str
    is_primary_candidate: bool
    is_default: bool
    enabled: bool
    device_requirements: str
    engineering_metrics: Optional[EngineeringMetricsSchema] = None

class DeviceStatusSchema(BaseModel):
    device_name: str
    device_type: str
    is_cuda_available: bool
    gpu_count: int
    memory_allocated_mib: Optional[float] = None
    memory_reserved_mib: Optional[float] = None

class ImageValidationResponse(BaseModel):
    valid: bool
    is_dfu_domain: bool = False
    domain_decision: str = ""
    dfu_score: Optional[float] = None
    threshold: Optional[float] = None
    top_detected_category: Optional[str] = None
    width: int
    height: int
    aspect_ratio: float
    format: str
    channels: int
    message: str

class DimensionSchema(BaseModel):
    width: int
    height: int

class InferenceTimingSchema(BaseModel):
    inference_latency_ms: float
    end_to_end_latency_ms: float
    positive_pixels: int
    ulcer_area_percentage: float

class ProvenanceRecordSchema(BaseModel):
    image_identifier: str
    filename: str
    source_category: str
    model_architecture: str
    condition: str
    seed: int
    threshold: float
    input_resolution: str
    model_sha256: str
    device: str
    export_type: str
    timestamp: str
    inference_latency_ms: float
    end_to_end_latency_ms: float
    certification_status: str

class InferenceResponseSchema(BaseModel):
    image_id: str
    original_filename: str
    source_category: str
    model_id: str
    architecture: str
    reader_facing_label: str
    condition: str
    seed: int
    input_resolution: str
    threshold: float
    threshold_status: str
    device: str
    export_format: str
    model_sha256: str
    certification_badge: str
    original_dimensions: DimensionSchema
    mask_dimensions: DimensionSchema
    original_image_b64: str
    ground_truth_b64: Optional[str] = None
    predicted_mask_b64: str
    overlay_b64: str
    probability_heatmap_b64: Optional[str] = None
    timing: InferenceTimingSchema
    export_id: str
    provenance_record: ProvenanceRecordSchema

class DemoSampleSchema(BaseModel):
    id: str
    name: str
    source_category: str
    has_ground_truth: bool
    description: str

class ExportDownloadInfo(BaseModel):
    export_id: str
    predicted_mask_url: str
    overlay_url: str
    visualization_url: str
    metadata_url: str
    timestamp: str

class HealthResponseSchema(BaseModel):
    status: str
    timestamp: str
    active_model_id: str
    active_architecture: str
    device: str
    threshold: float
    threshold_status: str
