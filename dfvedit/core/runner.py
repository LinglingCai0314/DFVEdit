"""
Main runner for video editing.

This module provides the main entry point for running video editing
with DFVEdit.
"""

from typing import Optional, Set
import time
import os

import torch
from diffusers.utils import export_to_video, load_video

from dfvedit.config.schema import DFVEditConfig
from dfvedit.core.pipeline_factory import build_pipe
from dfvedit.samplers.dfv_sampler import DFVSampler
from dfvedit.text.t5_embed import encode_prompt
from dfvedit.video.mask import process_mask_video, apply_mask_to_grad
from dfvedit.utils.logging import get_logger, print_config_summary
from dfvedit.utils.seed import set_seed
from dfvedit.utils.misc import ensure_dir, safe_copy

logger = get_logger()


def run_edit(
    config: DFVEditConfig,
    pipe=None,
) -> float:
    """
    Run video editing with the given configuration.

    Args:
        config: DFVEditConfig instance with all settings
        pipe: Optional pre-built pipeline (will be built if None)

    Returns:
        Total editing time in seconds
    """
    # Setup
    set_seed(config.seed)
    print_config_summary(config)
    device = torch.device(config.device)

    # Build pipeline if not provided
    if pipe is None:
        pipe, _ = build_pipe(config, device)

    # Create output directory
    output_dir = ensure_dir(config.output)

    # Copy config file to output
    safe_copy(getattr(config, '_config_path', None), output_dir)

    # Run the optimization
    start_time = time.time()
    _run_optimization(pipe, config, device, output_dir)
    end_time = time.time()

    elapsed = end_time - start_time
    logger.info(f"Editing time: {elapsed:.2f} seconds")

    return elapsed


def _run_optimization(
    pipe,
    config: DFVEditConfig,
    device: torch.device,
    output_dir,
) -> None:
    """
    Internal function to run the optimization loop.
    """
    # Extract config values
    video_cfg = config.video
    editing_cfg = config.editing

    # Load video
    logger.info(f"Loading video from {config.input}")
    input_video = load_video(config.input)
    logger.info(f"Original video: {len(input_video)} frames")

    # Apply frame selection
    start = video_cfg.start_frame
    num = video_cfg.num_frames
    rate = video_cfg.fps
    input_video = input_video[start:(num + start + 1) * rate:rate]
    logger.info(f"Selected: {len(input_video)} frames")

    # Prepare embeddings
    negative_prompt = _get_default_negative_prompt()

    # Match legacy memory behavior: preprocessing, text encoding and VAE encode
    # are all inference-only and should not build autograd graphs.
    with torch.no_grad():
        # Encode prompts
        logger.info("Encoding prompts...")
        source_embeds, source_neg_embeds = encode_prompt(
            pipe=pipe,
            prompt=editing_cfg.prompt_original,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=device,
            token_amplify_config=editing_cfg.token_amplify,
            debug_tokens=config.debug_tokens,
        )
        embedding_src = torch.cat([source_neg_embeds, source_embeds], dim=0)

        target_embeds, target_neg_embeds = encode_prompt(
            pipe=pipe,
            prompt=editing_cfg.prompt_target,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=device,
            token_amplify_config=None,  # Only amplify source
            debug_tokens=False,
        )
        embedding_tgt = torch.cat([target_neg_embeds, target_embeds], dim=0)

        # Prepare latents
        logger.info("Encoding video to latents...")
        video = pipe.video_processor.preprocess_video(
            input_video,
            height=video_cfg.height,
            width=video_cfg.width
        )
        video = video.to(device=device, dtype=pipe.vae.dtype)

        latents_src = pipe.vae.encode(video)['latent_dist'].mean

        # Normalize latents
        latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents_src.device, latents_src.dtype)
        latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latents_src.device, latents_src.dtype)
        latents_src = (latents_src - latents_mean) * latents_std

        logger.info(f"Source latents shape: {latents_src.shape}")

        # Process mask if provided
        mask = None
        if config.mask:
            logger.info("Processing mask...")
            mask_frames = load_video(config.mask)
            mask_frames = mask_frames[start:(num + start + 1) * rate:rate]
            mask = process_mask_video(
                mask_frames,
                pipe,
                video_cfg.height,
                video_cfg.width,
                device,
                debug_dir=output_dir if config.debug else None,
            )

    # Setup scheduler
    num_steps = editing_cfg.num_inference_steps
    pipe.scheduler.set_timesteps(num_steps, device=device)

    # Initialize sampler
    sampler = DFVSampler(pipe=pipe, device=device, config=editing_cfg)
    latents_edit = latents_src.clone()

    # Determine save steps
    save_steps = _get_save_steps(num_steps, editing_cfg.save.every, editing_cfg.save.steps)

    # Main optimization loop
    logger.info(f"Starting optimization for {num_steps - 1} steps...")
    for i in range(num_steps - 1):
        step_index = i + 1

        # Compute CDFV
        cdfv, _ = sampler.compute_cdfv(
            latents_src=latents_src,
            latents_edit=latents_edit,
            text_emb_src=embedding_src,
            text_emb_tgt=embedding_tgt,
            step_index=step_index,
            guidance_scale_src=editing_cfg.guidance_scale_source,
            guidance_scale_tgt=editing_cfg.guidance_scale_target,
        )

        # Apply mask after threshold
        if mask is not None and i > editing_cfg.mask_apply_after_step:
            cdfv = apply_mask_to_grad(cdfv, mask)

        # Update latents
        latents_edit = latents_edit + cdfv.to(latents_edit.dtype)

        # Save intermediate results
        if step_index in save_steps:
            _save_intermediate(
                latents_edit,
                pipe,
                latents_mean,
                latents_std,
                output_dir,
                step_index,
            )

    logger.info("Optimization complete!")


def _get_default_negative_prompt() -> str:
    """Get the default negative prompt."""
    return (
        "Bright tones, overexposed, static, blurred details, subtitles, "
        "style, works, paintings, images, static, overall gray, worst quality, "
        "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
        "fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"
    )


def _get_save_steps(num_steps: int, save_every: int, save_list: Optional[list]) -> Set[int]:
    """Determine which steps to save."""
    max_step = max(num_steps - 1, 1)
    steps: Set[int] = {1, max_step}

    if isinstance(save_every, int) and save_every > 0:
        steps.update(range(save_every, max_step + 1, save_every))

    if save_list:
        for raw_step in save_list:
            try:
                step = int(raw_step)
            except (TypeError, ValueError):
                continue

            # Backward compatibility: old examples used 0 to mean "first save".
            if step == 0:
                step = 1
            if 1 <= step <= max_step:
                steps.add(step)

    return steps


def _save_intermediate(
    latents,
    pipe,
    latents_mean,
    latents_std,
    output_dir,
    step_index,
) -> None:
    """Save intermediate video result."""
    with torch.no_grad():
        # Denormalize latents
        z = latents.clone().detach()
        z = z / latents_std + latents_mean

        # Decode
        video = pipe.vae.decode(z, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(
            video=video.detach(),
            output_type="np"
        )[0]

        # Export
        output_path = os.path.join(output_dir, f"video_{step_index}.mp4")
        try:
            export_to_video(video, output_path, fps=10)
            logger.debug(f"Saved intermediate video to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save video: {e}")
