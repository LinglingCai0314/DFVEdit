"""
Schedule utilities for timestep and sigma management.

Provides functions for handling timesteps, sigmas, and step indices
in diffusion sampling schedules.
"""

from typing import Tuple, Optional

import torch


def get_sigma_at_step(
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """
    Get sigma value at a specific step index.

    Args:
        sigmas: Sigma schedule tensor
        step_index: Current step index

    Returns:
        Sigma value at the step
    """
    return sigmas[step_index].flatten()


def get_sigma_pair(
    sigmas: torch.Tensor,
    step_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get current and next sigma values.

    Args:
        sigmas: Sigma schedule tensor
        step_index: Current step index

    Returns:
        Tuple of (sigma_t, sigma_t_next)
    """
    sigma_t = sigmas[step_index].flatten()
    sigma_t_next = sigmas[step_index + 1].flatten()
    return sigma_t, sigma_t_next


def compute_delta_sigma(
    sigma_t: torch.Tensor,
    sigma_t_next: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the delta between consecutive sigmas.

    This represents the step size in the diffusion process.

    Args:
        sigma_t: Current sigma
        sigma_t_next: Next sigma

    Returns:
        Delta sigma (step size)
    """
    return sigma_t_next - sigma_t


def get_timestep_at_step(
    timesteps: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """
    Get timestep value at a specific step index.

    Args:
        timesteps: Timestep schedule tensor
        step_index: Current step index

    Returns:
        Timestep value (scalar tensor)
    """
    return timesteps[step_index]


def add_noise_at_step(
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    noise: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to latents at a specific sigma level.

    Implements the noising process: z_t = sigma_t * noise + (1 - sigma_t) * z

    Args:
        latents: Clean latent tensor
        sigmas: Sigma schedule
        step_index: Current step index
        noise: Optional pre-generated noise tensor

    Returns:
        Tuple of (noised_latents, noise)
    """
    if noise is None:
        noise = torch.randn_like(latents)

    sigma_t = get_sigma_at_step(sigmas, step_index)
    sigma_t = sigma_t.to(latents.device)

    noised_latents = sigma_t * noise + (1 - sigma_t) * latents

    return noised_latents, noise
