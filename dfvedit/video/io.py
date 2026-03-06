"""
Video I/O utilities.

Provides functions for loading and exporting video files.
"""

from typing import List, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image

from dfvedit.utils.logging import get_logger

logger = get_logger()


def load_video_frames(
    video_path: Union[str, Path],
    start_frame: int = 0,
    num_frames: int = -1,
    fps: int = 1,
) -> List[Image.Image]:
    """
    Load video frames as PIL images.

    Args:
        video_path: Path to video file
        start_frame: Starting frame index
        num_frames: Number of frames to load (-1 for all)
        fps: Frame sampling rate (1 = every frame, 2 = every other frame)

    Returns:
        List of PIL Image frames
    """
    try:
        from diffusers.utils import load_video
    except ImportError:
        raise ImportError("diffusers is required for video loading")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Load all frames
    frames = load_video(str(video_path))
    logger.info(f"Loaded {len(frames)} frames from {video_path}")

    # Apply frame selection
    if start_frame > 0 or num_frames > 0 or fps > 1:
        end_frame = (num_frames + start_frame + 1) * fps if num_frames > 0 else len(frames)
        frames = frames[start_frame:end_frame:fps]
        logger.info(f"Selected {len(frames)} frames (start={start_frame}, num={num_frames}, fps={fps})")

    return frames


def export_video(
    video: Union[np.ndarray, List[np.ndarray]],
    output_path: Union[str, Path],
    fps: int = 10,
) -> Path:
    """
    Export video frames to a video file.

    Args:
        video: Video frames as numpy array [T, H, W, C] or list of frames
        output_path: Output video path
        fps: Frames per second

    Returns:
        Path to the exported video
    """
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        raise ImportError("diffusers is required for video export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        export_to_video(video, str(output_path), fps=fps)
        logger.info(f"Exported video to {output_path}")
    except Exception as e:
        logger.error(f"Failed to export video to {output_path}: {e}")
        raise

    return output_path
