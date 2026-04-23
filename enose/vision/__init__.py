"""Florence-2 VLM for open-vocabulary object detection."""

from .gpu import check_gpu_availability
from .florence import load_florence_model, process_image_with_vlm

__all__ = ["check_gpu_availability", "load_florence_model", "process_image_with_vlm"]
