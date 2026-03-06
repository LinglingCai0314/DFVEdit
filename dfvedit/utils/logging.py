"""
Logging utilities for DFVEdit.

Provides a centralized logging system with pretty printing for configurations.
"""

import logging
import sys
from typing import Any

# Module-level logger
_logger: logging.Logger = logging.getLogger("dfvedit")


def setup_logging(
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream: Any = None,
) -> None:
    """
    Setup logging configuration for DFVEdit.

    Args:
        level: Logging level (default: INFO)
        format_string: Log format string
        stream: Output stream (default: sys.stderr)
    """
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(format_string))

    _logger.setLevel(level)
    _logger.addHandler(handler)


def get_logger() -> logging.Logger:
    """Get the DFVEdit logger instance."""
    return _logger


def print_config_summary(config: Any) -> None:
    """
    Print a pretty summary of the configuration.

    Args:
        config: DFVEditConfig instance
    """
    from dfvedit.config.schema import DFVEditConfig

    if not isinstance(config, DFVEditConfig):
        _logger.warning("Cannot print config summary: not a DFVEditConfig instance")
        return

    separator = "=" * 60
    lines = [
        separator,
        "DFVEdit Configuration",
        separator,
        f"Input:           {config.input}",
        f"Output:          {config.output}",
        f"Mask:            {config.mask or '(none)'}",
        f"Model:           {config.model.name} ({config.model.dtype})",
        f"Prompt Original: {config.editing.prompt_original}",
        f"Prompt Target:   {config.editing.prompt_target}",
        f"Video:           {config.video.width}x{config.video.height}, {config.video.num_frames} frames @ {config.video.fps} fps",
        f"Steps:           {config.editing.num_inference_steps}",
        f"Guidance:        {config.editing.guidance_scale_source} / {config.editing.guidance_scale_target}",
        f"Seed:            {config.seed}",
        f"Device:          {config.device}",
    ]

    # Token amplify info
    if config.editing.token_amplify.enabled and config.editing.token_amplify.words:
        lines.append(f"Token Amplify:   {config.editing.token_amplify.words} @ {config.editing.token_amplify.amplitude}x")

    lines.append(separator)

    for line in lines:
        print(line)
