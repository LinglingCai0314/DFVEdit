"""Backward-compatible aliases for prompt cleaning helpers."""

from dfvedit.text.clean import basic_clean, prompt_clean, whitespace_clean

__all__ = ["prompt_clean", "basic_clean", "whitespace_clean"]
