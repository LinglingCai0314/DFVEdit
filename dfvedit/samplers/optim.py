"""
Custom SGD optimizer implementation.

Provides a custom SGD optimizer with momentum support for
latent optimization in DFVEdit.
"""

from typing import List, Optional
import torch


class SGD:
    """
    Custom SGD optimizer with momentum support.

    This is a minimal implementation for use in DFVEdit's
    latent optimization process.
    """

    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """
        Initialize SGD optimizer.

        Args:
            params: List of parameters to optimize
            lr: Learning rate (default: 1e-3)
            momentum: Momentum factor (default: 0)
            dampening: Dampening for momentum (default: 0)
            weight_decay: L2 regularization (default: 0)
            nesterov: Use Nesterov momentum (default: False)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening = 0")

        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Momentum buffers
        self.momentum_buffers: List[Optional[torch.Tensor]] = [None] * len(params)

    def step(self) -> None:
        """Perform a single optimization step."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Momentum
            if self.momentum != 0:
                buf = self.momentum_buffers[i]

                if buf is None:
                    buf = torch.clone(grad).detach()
                    self.momentum_buffers[i] = buf
                else:
                    buf.mul_(self.momentum).add_(grad, alpha=1 - self.dampening)

                # Nesterov momentum
                if self.nesterov:
                    grad = grad.add(buf, alpha=self.momentum)
                else:
                    grad = buf

            # Update parameters
            param.data.add_(grad, alpha=-self.lr)

    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
