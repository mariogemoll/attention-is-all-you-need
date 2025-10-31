"""Utility functions for model operations."""

import torch


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_parameter_count(total_params: int, trainable_params: int) -> str:
    """Format parameter counts in a human-readable format."""

    def format_number(num: int) -> str:
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)

    total_str = format_number(total_params)
    trainable_str = format_number(trainable_params)

    if total_params == trainable_params:
        return f"Total parameters: {total_str} ({total_params:,})"
    else:
        return (
            f"Total parameters: {total_str} ({total_params:,}), Trainable: {trainable_str} "
            f"({trainable_params:,})"
        )


def print_model_parameters(model: torch.nn.Module) -> None:
    """Print model parameter count in a formatted way."""
    total_params, trainable_params = count_parameters(model)
    param_info = format_parameter_count(total_params, trainable_params)
    print(f"Model created: {param_info}")
