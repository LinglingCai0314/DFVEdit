"""
Type definitions for DFVEdit.

Provides type aliases and protocol definitions used throughout the codebase.
"""

from typing import Tuple, Union, List, Optional
import torch

# Type aliases for tensors (paper-style naming)
T = torch.Tensor
TN = Optional[torch.Tensor]
TS = Union[Tuple[T, ...], List[T]]

# Latent naming (paper-style)
Latents = torch.Tensor  # [B, C, T, H, W]
LatentsSrc = torch.Tensor
LatentsEdit = torch.Tensor

# Velocity / Flow
Velocity = torch.Tensor
CDFV = torch.Tensor  # Conditional Delta Flow Vector

# Embeddings
TextEmbedding = torch.Tensor  # [B, Seq, Hidden]

# Schedule
Sigma = torch.Tensor
Timestep = torch.Tensor
StepIndex = int


# Default device
def get_default_device() -> torch.device:
    """Get the default device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
