"""
Video preprocessing utilities.

Wrappers around diffusers video processor for preprocessing
and postprocessing video tensors.
"""

from typing import List, Union, Optional
import torch
from PIL import Image


def preprocess_video(
    frames: List[Image.Image],
    height: int,
    width: int,
    processor,
) -> torch.Tensor:
    """
    Preprocess video frames for the model.

    Args:
        frames: List of PIL Image frames
        height: Target height
        width: Target width
        processor: Video processor from pipeline

    Returns:
        Preprocessed video tensor
    """
    video = processor.preprocess_video(frames, height=height, width=width)
    return video


def postprocess_video(
    video_tensor: torch.Tensor,
    processor,
    output_type: str = "np",
) -> List:
    """
    Postprocess video tensor to frames.

    Args:
        video_tensor: Video tensor from model
        processor: Video processor from pipeline
        output_type: Output type ("np" for numpy, "pil" for PIL images)

    Returns:
        List of video frames
    """
    video = processor.postprocess_video(video=video_tensor.detach(), output_type=output_type)
    return video[0] if isinstance(video, list) and len(video) == 1 else video
