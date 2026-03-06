"""
T5 text encoder utilities for Wan model.

Provides functions for encoding prompts using T5 text encoder
with support for token amplification.
"""

from typing import Union, List, Optional

import torch

from dfvedit.text.clean import prompt_clean
from dfvedit.text.token_amp import TokenAmplifier
from dfvedit.config.schema import TokenAmplifyConfig
from dfvedit.utils.logging import get_logger

logger = get_logger()


def _get_t5_prompt_embeds(
    pipe,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    token_amplifier: Optional[TokenAmplifier] = None,
    debug_tokens: bool = False,
) -> torch.Tensor:
    """
    Get T5 prompt embeddings with optional token amplification.

    Args:
        pipe: Pipeline with text encoder and tokenizer
        prompt: Text prompt(s)
        num_videos_per_prompt: Number of videos to generate per prompt
        max_sequence_length: Maximum sequence length for tokenization
        device: Target device
        dtype: Target dtype
        token_amplifier: TokenAmplifier instance for embedding amplification
        debug_tokens: Whether to print token debug info

    Returns:
        Prompt embedding tensor
    """
    device = device or pipe._execution_device
    dtype = dtype or pipe.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(p) for p in prompt]
    batch_size = len(prompt)

    # Tokenize
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask

    # Get tokens for amplification
    tokens = pipe.tokenizer.convert_ids_to_tokens(text_input_ids[0])

    if debug_tokens:
        logger.debug(f"Tokens: {tokens}")

    # Encode
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = pipe.text_encoder(
        text_input_ids.to(device),
        mask.to(device)
    ).last_hidden_state

    # Apply token amplification if configured
    if token_amplifier is not None:
        prompt_embeds = token_amplifier.apply(
            prompt_embeds,
            tokens,
            debug=debug_tokens
        )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # Truncate to actual sequence length and pad
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
         for u in prompt_embeds],
        dim=0
    )

    # Duplicate for each video per prompt
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    pipe,
    prompt: Union[str, List[str]],
    negative_prompt: Optional[Union[str, List[str]]] = None,
    do_classifier_free_guidance: bool = True,
    num_videos_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    token_amplify_config: Optional[TokenAmplifyConfig] = None,
    debug_tokens: bool = False,
) -> tuple:
    """
    Encode prompt with optional negative prompt and token amplification.

    Args:
        pipe: Pipeline with text encoder and tokenizer
        prompt: Positive prompt(s)
        negative_prompt: Negative prompt(s)
        do_classifier_free_guidance: Whether to use CFG
        num_videos_per_prompt: Number of videos per prompt
        prompt_embeds: Pre-computed positive embeddings (optional)
        negative_prompt_embeds: Pre-computed negative embeddings (optional)
        max_sequence_length: Maximum sequence length
        device: Target device
        dtype: Target dtype
        token_amplify_config: TokenAmplifyConfig for embedding amplification
        debug_tokens: Whether to print token debug info

    Returns:
        Tuple of (prompt_embeds, negative_prompt_embeds)
    """
    device = device or pipe._execution_device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt is not None else prompt_embeds.shape[0]

    # Create token amplifier if configured
    token_amplifier = None
    if token_amplify_config is not None and token_amplify_config.enabled:
        token_amplifier = TokenAmplifier(token_amplify_config)

    # Encode positive prompt
    if prompt_embeds is None:
        prompt_embeds = _get_t5_prompt_embeds(
            pipe=pipe,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            token_amplifier=token_amplifier,
            debug_tokens=debug_tokens,
        )

    # Encode negative prompt if using CFG
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        negative_prompt = negative_prompt or ""
        negative_prompt = (
            batch_size * [negative_prompt]
            if isinstance(negative_prompt, str)
            else negative_prompt
        )

        # Note: Token amplification typically only applied to positive prompt
        negative_prompt_embeds = _get_t5_prompt_embeds(
            pipe=pipe,
            prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
            token_amplifier=None,  # No amplification for negative prompt
            debug_tokens=False,
        )

    return prompt_embeds, negative_prompt_embeds
