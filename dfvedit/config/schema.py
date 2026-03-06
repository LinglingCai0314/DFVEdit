"""
Configuration schemas for DFVEdit.

This module defines the dataclasses that represent the configuration
structure for video/image editing tasks.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class TokenAmplifyConfig:
    """Configuration for token amplitude amplification.

    This allows boosting the embedding of specific tokens during
    prompt encoding to emphasize certain words in the edit.
    """
    words: List[str] = field(default_factory=list)
    amplitude: float = 1.0
    enabled: bool = True


@dataclass
class SaveConfig:
    """Configuration for saving intermediate results."""
    every: int = 5  # Save every N steps
    steps: Optional[List[int]] = None  # Specific steps to save (optional)


@dataclass
class VideoConfig:
    """Video processing configuration."""
    height: int = 512
    width: int = 512
    num_frames: int = 28
    start_frame: int = 0
    fps: int = 8


@dataclass
class EditingConfig:
    """Editing parameters configuration."""
    prompt_original: str = ""
    prompt_target: str = ""
    num_inference_steps: int = 50
    guidance_scale_source: float = 5.0
    guidance_scale_target: float = 15.0
    amplitude: float = 1.0
    use_dds: bool = True

    # Token amplification (new feature)
    token_amplify: TokenAmplifyConfig = field(default_factory=TokenAmplifyConfig)

    # Save configuration (new feature)
    save: SaveConfig = field(default_factory=SaveConfig)

    # Mask application threshold
    mask_apply_after_step: int = 100


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "wanx"
    path: str = ""
    dtype: str = "bfloat16"


@dataclass
class DFVEditConfig:
    """Unified configuration for DFVEdit.

    This is the main configuration class that aggregates all
    sub-configurations for a video editing task.
    """
    # Required fields
    input: str = ""
    output: str = ""

    # Mask path (optional)
    mask: str = ""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    editing: EditingConfig = field(default_factory=EditingConfig)

    # System settings
    seed: int = 42
    device: str = "cuda"

    # Debug settings
    debug: bool = False
    debug_tokens: bool = False

    # Legacy support (for backward compatibility)
    dataset_config: Optional[Dict] = None
    editing_config: Optional[Dict] = None

    def is_legacy_format(self) -> bool:
        """Check if this is a legacy config format."""
        return self.dataset_config is not None
