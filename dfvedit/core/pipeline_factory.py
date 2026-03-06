"""
Pipeline factory for creating diffusion pipelines.

Provides functions for building and configuring diffusion pipelines
for different models (Wan, CogVideoX, etc.).
"""

from typing import Optional, Tuple
import torch

from dfvedit.config.schema import DFVEditConfig, ModelConfig
from dfvedit.utils.logging import get_logger

logger = get_logger()


def build_wan_pipe(
    config: DFVEditConfig,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Build a Wan2.1 video-to-video pipeline.

    Args:
        config: DFVEditConfig with model settings
        device: Target device (default: from config)

    Returns:
        Tuple of (pipeline, scheduler)
    """
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers import FlowMatchEulerDiscreteScheduler

    device = device or torch.device(config.device)
    model_path = config.model.path

    if not model_path:
        raise ValueError("Model path not configured. Set model.path in config or CKPT_ROOT env var.")

    logger.info(f"Loading Wan model from {model_path}")

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype, torch.bfloat16)
    vae_dtype = torch.float16  # VAE always uses float16

    # Load VAE
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=vae_dtype
    )

    # Load pipeline
    pipe = WanVideoToVideoPipeline.from_pretrained(
        model_path,
        vae=vae,
        torch_dtype=dtype
    )

    # Setup scheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    pipe.scheduler = scheduler

    # Move to device
    pipe.to(device)

    logger.info(f"Pipeline loaded with dtype={dtype}, device={device}")

    return pipe, scheduler


def build_cogvideox_pipe(
    config: DFVEditConfig,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Build a CogVideoX video-to-video pipeline.

    Args:
        config: DFVEditConfig with model settings
        device: Target device

    Returns:
        Tuple of (pipeline, scheduler)
    """
    from diffusers import CogVideoXVideoToVideoPipeline

    device = device or torch.device(config.device)
    model_path = config.model.path

    if not model_path:
        raise ValueError("Model path not configured.")

    logger.info(f"Loading CogVideoX model from {model_path}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype
    )
    pipe.to(device)

    return pipe, pipe.scheduler


def build_pipe(
    config: DFVEditConfig,
    device: Optional[torch.device] = None,
) -> Tuple:
    """
    Build a pipeline based on model name.

    Args:
        config: DFVEditConfig with model settings
        device: Target device

    Returns:
        Tuple of (pipeline, scheduler)
    """
    model_builders = {
        "wanx": build_wan_pipe,
        "wan": build_wan_pipe,
        "cogvideox": build_cogvideox_pipe,
    }

    model_name = config.model.name.lower()
    builder = model_builders.get(model_name)

    if builder is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_builders.keys())}")

    return builder(config, device)
