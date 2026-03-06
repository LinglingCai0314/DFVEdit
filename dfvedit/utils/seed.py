"""
Seed utilities for reproducibility.
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Seed value. If None, a random seed will be generated.
        deterministic: If True, enables deterministic mode in PyTorch (may impact performance)

    Returns:
        The seed that was set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic mode
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
