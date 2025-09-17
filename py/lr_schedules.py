from __future__ import annotations

import math
from typing import Callable


def aiayn_lr(step_num: int, d_model: int, warmup_steps: int) -> float:
    """
    Learning rate schedule from "Attention is All You Need".

    Args:
        step_num: Current training step (1-based).
        d_model: Model dimensionality.
        warmup_steps: Number of warmup steps.

    Returns:
        The learning rate at the given step.
    """
    if step_num <= 0:
        raise ValueError("step_num must be >= 1")
    if d_model <= 0:
        raise ValueError("d_model must be >= 1")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be >= 1")

    scale = d_model**-0.5
    return float(scale * min(step_num**-0.5, step_num * (warmup_steps**-1.5)))


def cosine_lr(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """
    Returns a function(step) -> multiplier in [0, 1] for LambdaLR.

    Args:
        total_steps: total number of training steps
        warmup_steps: number of warmup steps
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1")
    if not (0 <= warmup_steps <= total_steps):
        raise ValueError("0 <= warmup_steps <= total_steps")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

    return lr_lambda
