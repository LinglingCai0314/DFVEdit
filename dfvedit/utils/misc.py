"""
Miscellaneous utility functions.
"""

import os
import shutil
from pathlib import Path
from typing import Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """
    Safely copy a file to a destination.

    Args:
        src: Source file path
        dst: Destination file or directory path

    Returns:
        Path object for the destination file
    """
    src = Path(src)
    dst = Path(dst)

    # If dst is a directory, use the source filename
    if dst.is_dir():
        dst = dst / src.name

    shutil.copy2(src, dst)
    return dst


def get_file_stem(path: Union[str, Path]) -> str:
    """Get the stem (filename without extension) of a path."""
    return Path(path).stem


def get_file_suffix(path: Union[str, Path]) -> str:
    """Get the suffix (extension) of a path."""
    return Path(path).suffix
