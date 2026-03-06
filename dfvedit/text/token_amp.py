"""
Token amplification utilities.

Provides configurable token embedding amplification to emphasize
specific words during prompt encoding.
"""

from typing import List, Optional

import torch

from dfvedit.config.schema import TokenAmplifyConfig
from dfvedit.utils.logging import get_logger

logger = get_logger()


class TokenAmplifier:
    """
    Configurable token embedding amplifier.

    This class allows boosting the embedding values of specific tokens
    to emphasize certain words in the editing process.

    Example:
        config = TokenAmplifyConfig(
            words=['▁white', '▁Polar', '▁bear'],
            amplitude=2.0
        )
        amplifier = TokenAmplifier(config)
        amplified_embeds = amplifier.apply(prompt_embeds, tokens)
    """

    def __init__(self, config: Optional[TokenAmplifyConfig] = None):
        """
        Initialize the token amplifier.

        Args:
            config: TokenAmplifyConfig instance (optional)
        """
        self.config = config or TokenAmplifyConfig()

    def apply(
        self,
        prompt_embeds: torch.Tensor,
        tokens: List[str],
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Apply token amplification to prompt embeddings.

        Args:
            prompt_embeds: Prompt embedding tensor [batch, seq_len, hidden_dim]
            tokens: List of tokens corresponding to the embedding sequence
            debug: Whether to print debug information

        Returns:
            Amplified prompt embeddings
        """
        if not self.config.enabled or not self.config.words:
            return prompt_embeds

        amplitude = self.config.amplitude
        target_words = self.config.words

        if debug:
            logger.debug(f"Tokens: {tokens}")
            logger.debug(f"Target words: {target_words}")
            logger.debug(f"Amplitude: {amplitude}")

        # Create amplitude tensor (default 1.0)
        seq_len = prompt_embeds.shape[1]
        amplitude_tensor = torch.ones(seq_len, dtype=prompt_embeds.dtype, device=prompt_embeds.device)

        # Apply amplification to matching tokens
        for word in target_words:
            for i, token in enumerate(tokens):
                if token == word:
                    amplitude_tensor[i] = amplitude
                    if debug:
                        logger.debug(f"Amplifying token '{token}' at position {i} to {amplitude}")

        # Apply to embeddings
        # prompt_embeds shape: [batch, seq_len, hidden_dim]
        amplitude_tensor = amplitude_tensor.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        prompt_embeds = prompt_embeds * amplitude_tensor

        return prompt_embeds

    def update_config(self, config: TokenAmplifyConfig) -> None:
        """Update the amplifier configuration."""
        self.config = config

    def set_words(self, words: List[str]) -> None:
        """Set the target words for amplification."""
        self.config.words = words

    def set_amplitude(self, amplitude: float) -> None:
        """Set the amplification amplitude."""
        self.config.amplitude = amplitude
