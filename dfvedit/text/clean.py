"""
Prompt cleaning utilities.

Provides functions for cleaning and normalizing text prompts
before tokenization and embedding.
"""

import ftfy
import html
import regex as re


def whitespace_clean(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def basic_clean(text: str) -> str:
    """
    Basic text cleaning using ftfy.

    Fixes encoding issues and unescapes HTML entities.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def prompt_clean(text: str) -> str:
    """
    Full prompt cleaning pipeline.

    Applies both basic cleaning and whitespace normalization.

    Args:
        text: Input prompt

    Returns:
        Cleaned prompt
    """
    text = whitespace_clean(basic_clean(text))
    return text
