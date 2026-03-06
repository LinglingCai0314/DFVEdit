"""
DFV (Delta Flow Vector) Sampler implementation.

This module implements the core sampling logic for DFVEdit,
including the CDFV (Conditional Delta Flow Vector) computation.
"""

from typing import Tuple, Optional, Any

import torch

from dfvedit.samplers.schedules import get_sigma_pair, compute_delta_sigma
from dfvedit.utils.logging import get_logger

logger = get_logger()

# Type aliases
T = torch.Tensor


class DFVSampler:
    """
    DFV Sampler for video editing.

    Implements the Conditional Delta Flow Vector (CDFV) computation
    for zero-shot video editing on diffusion transformers.
    """

    def __init__(
        self,
        pipe,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[Any] = None,
    ):
        """
        Initialize the DFV sampler.

        Args:
            pipe: Diffusion pipeline with transformer and scheduler
            device: Device to run on
            dtype: Data type for computations
            config: Optional configuration object
        """
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        self.config = config

        # Initialize transformer (freeze parameters)
        self.transformer = pipe.transformer
        for p in self.transformer.parameters():
            p.requires_grad = False

        # Get sigma schedule from scheduler
        with torch.inference_mode():
            self.sigmas = pipe.scheduler.sigmas
            logger.debug(f"Scheduler sigmas: {self.sigmas}")

    def add_noise_at_step(
        self,
        latents: T,
        noise: Optional[T] = None,
        step_index: int = 0,
    ) -> Tuple[T, T, T]:
        """
        Add noise to latents at the specified step.

        Args:
            latents: Clean latent tensor
            noise: Optional pre-generated noise
            step_index: Current step index

        Returns:
            Tuple of (noised_latents, noise, sigma_t)
        """
        if noise is None:
            noise = torch.randn_like(latents)

        sigma_t, _ = get_sigma_pair(self.sigmas, step_index)
        sigma_t = sigma_t.to(latents.device)

        noised_latents = sigma_t * noise + (1 - sigma_t) * latents

        return noised_latents, noise, sigma_t

    def predict_velocity(
        self,
        latents_noised: T,
        timestep: T,
        text_embeddings: T,
        sigma_t: T,
        sigma_t_next: T,
        guidance_scale: float = 7.5,
    ) -> Tuple[T, T]:
        """
        Predict velocity (noise prediction) for the denoising step.

        Args:
            latents_noised: Noised latent tensor
            timestep: Current timestep
            text_embeddings: Text conditioning embeddings
            sigma_t: Current sigma
            sigma_t_next: Next sigma
            guidance_scale: CFG scale

        Returns:
            Tuple of (velocity, pred_z0)
        """
        # Prepare inputs (duplicate for CFG)
        latent_input = torch.cat([latents_noised] * 2)
        timestep = torch.cat([timestep] * 2)

        # Transformer forward pass
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noise_pred = self.transformer(
                hidden_states=latent_input,
                encoder_hidden_states=text_embeddings,
                timestep=timestep,
                attention_kwargs=None,
                return_dict=False,
            )[0]
            velocity = noise_pred.float()

        # Apply CFG
        v_uncond, v_cond = velocity.chunk(2)
        velocity = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Compute delta sigma and scale velocity
        delta_sigma = compute_delta_sigma(sigma_t, sigma_t_next)
        velocity = delta_sigma * velocity

        assert torch.isfinite(velocity).all(), "Non-finite values in velocity prediction"

        # Predict z0
        pred_z0 = latents_noised + velocity

        return velocity, pred_z0

    def compute_cdfv(
        self,
        latents_src: T,
        latents_edit: T,
        text_emb_src: T,
        text_emb_tgt: T,
        step_index: int,
        guidance_scale_src: float = 7.5,
        guidance_scale_tgt: float = 7.5,
    ) -> Tuple[T, T]:
        """
        Compute the Conditional Delta Flow Vector (CDFV).

        This is the core operation that computes the editing direction
        by comparing the velocity predictions for source and target.

        Args:
            latents_src: Source video latents
            latents_edit: Editing video latents
            text_emb_src: Source text embeddings
            text_emb_tgt: Target text embeddings
            step_index: Current step index
            guidance_scale_src: Guidance scale for source
            guidance_scale_tgt: Guidance scale for target

        Returns:
            Tuple of (cdfv, aux_loss)
        """
        with torch.inference_mode():
            # Add noise to both latents
            latents_src_noised, noise, sigma_t, sigma_t_next = self._add_noise(
                latents_src, step_index
            )
            # Use the same noise realization for both branches to keep
            # the source/target velocity delta comparable.
            latents_edit_noised, _, _, _ = self._add_noise(
                latents_edit, step_index, noise=noise
            )

            # Get timestep
            timestep = self.pipe.scheduler.timesteps[step_index].unsqueeze(0)

            # Predict velocities
            v_src, _ = self.predict_velocity(
                latents_src_noised,
                timestep,
                text_emb_src,
                sigma_t,
                sigma_t_next,
                guidance_scale=guidance_scale_src,
            )

            v_tgt, _ = self.predict_velocity(
                latents_edit_noised,
                timestep,
                text_emb_tgt,
                sigma_t,
                sigma_t_next,
                guidance_scale=guidance_scale_tgt,
            )

            # CDFV is the difference between target and source velocities
            cdfv = v_tgt - v_src

        # Auxiliary loss for monitoring
        aux_loss = latents_edit * cdfv.clone()
        aux_loss = aux_loss.sum() / (latents_edit.shape[2] * latents_edit.shape[3])

        return cdfv, aux_loss

    def _add_noise(
        self,
        latents: T,
        step_index: int,
        noise: Optional[T] = None,
    ) -> Tuple[T, T, T, T]:
        """Add noise to a latent tensor and return sigma pair."""
        sigma_t, sigma_t_next = get_sigma_pair(self.sigmas, step_index)
        sigma_t = sigma_t.to(latents.device)
        sigma_t_next = sigma_t_next.to(latents.device)

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noised = sigma_t * noise + (1 - sigma_t) * latents

        return latents_noised, noise, sigma_t, sigma_t_next
