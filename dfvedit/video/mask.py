"""
Video mask processing utilities.

Provides functions for processing video masks and converting
them to latent space masks for selective editing.
"""

from typing import List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dfvedit.utils.logging import get_logger

logger = get_logger()


def process_mask_video(
    mask_frames: List[Image.Image],
    pipe,
    height: int,
    width: int,
    device: torch.device,
    threshold: float = 0.4,
    num_channels: int = 16,
    debug_dir: Optional[Path] = None,
) -> torch.Tensor:
    """
    Process mask video to latent space mask.

    Args:
        mask_frames: List of mask frame images
        pipe: Diffusion pipeline with VAE
        height: Target height
        width: Target width
        device: Device to run on
        threshold: Binarization threshold
        num_channels: Number of channels to repeat mask for
        debug_dir: Optional directory to save debug masks

    Returns:
        Binary mask tensor in latent space
    """
    # Preprocess mask frames
    mask = pipe.video_processor.preprocess_video(mask_frames, height=height, width=width)
    mask = mask.to(device=device, dtype=pipe.vae.dtype)

    # Encode to latent space
    latent_tensor = pipe.vae.encode(mask)['latent_dist'].mean

    # Normalize
    mean = latent_tensor.mean()
    std = latent_tensor.std()
    epsilon = 1e-6
    latent_tensor = (latent_tensor - mean) / (std + epsilon)

    # Scale to [0, 1]
    min_val = torch.min(latent_tensor)
    max_val = torch.max(latent_tensor)
    mask = (latent_tensor - min_val) / (max_val - min_val)

    # Binarize
    mask = torch.where(
        mask > threshold,
        torch.tensor(1.0, device=device),
        torch.tensor(0.0, device=device)
    )

    # Repeat for all channels
    mask = mask[0][0].repeat(1, num_channels, 1, 1, 1)
    mask = mask.to(device=device, dtype=pipe.vae.dtype)

    logger.info(f"Mask shape: {mask.shape}")

    # Debug: save mask frames
    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_mask_debug(mask, debug_dir)

    return mask


def save_mask_debug(
    mask: torch.Tensor,
    output_dir: Path,
    num_frames: int = 7,
) -> None:
    """
    Save mask frames for debugging.

    Args:
        mask: Mask tensor [B, C, T, H, W]
        output_dir: Output directory
        num_frames: Number of frames to save
    """
    for i in range(min(num_frames, mask.shape[2])):
        mask_np = mask[0][0][i].cpu().numpy()

        # Convert to 0-255 uint8
        mask_uint8 = (mask_np * 255).astype(np.uint8)
        image = Image.fromarray(mask_uint8)
        image = image.resize((512, 512), resample=Image.Resampling.NEAREST)

        img_path = output_dir / f"mask_frame_{i}.png"
        image.save(img_path)
        logger.debug(f"Saved mask frame to {img_path}")


def apply_mask_to_grad(
    grad: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Apply mask to gradient tensor.

    Args:
        grad: Gradient tensor
        mask: Binary mask tensor

    Returns:
        Masked gradient
    """
    return grad * mask
