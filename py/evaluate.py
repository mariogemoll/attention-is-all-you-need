#!/usr/bin/env python3
"""
Standalone evaluation script for model evaluation on validation dataset.

Usage:
    python evaluate.py <model_weights_file>
    python evaluate.py model_epoch_5.pt
    python evaluate.py checkpoints/model_latest.pt
"""

import argparse
import os
import sys

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from data import open_buckets
from model import Transformer
from params import pad
from training import evaluate
from util import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on validation dataset")
    parser.add_argument(
        "model_weights_file",
        help="Path to the model weights file (e.g., model_epoch_5.pt, checkpoints/model_latest.pt)",
    )
    parser.add_argument(
        "--dataset",
        default="../4_tokens/val",
        help="Path to validation dataset (default: ../4_tokens/val)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Skip TensorBoard logging",
    )

    args = parser.parse_args()

    # Check if model weights file exists
    if not os.path.exists(args.model_weights_file):
        print(f"Error: Model weights file not found: {args.model_weights_file}")
        sys.exit(1)

    # Check if dataset files exist (check for .bin and .idx files)
    bin_file = f"{args.dataset}.bin"
    idx_file = f"{args.dataset}.idx"
    if not (os.path.exists(bin_file) and os.path.exists(idx_file)):
        print("Error: Dataset not found. Expected files:")
        print(f"  {bin_file}")
        print(f"  {idx_file}")
        sys.exit(1)

    print(f"Loading model weights from: {args.model_weights_file}")
    print(f"Evaluating on dataset: {args.dataset}")

    # Initialize device and model
    device = get_device()
    model = Transformer().to(device)

    # Load model weights
    try:
        model_state_dict = torch.load(args.model_weights_file, map_location=device)
        model.load_state_dict(model_state_dict)
        print("✓ Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    # Initialize loss criterion
    criterion = CrossEntropyLoss(ignore_index=pad)

    # Initialize TensorBoard writer if requested
    writer = None if args.no_tensorboard else SummaryWriter()  # type: ignore

    # Run evaluation
    try:
        with open_buckets(args.dataset) as valset:
            print("Starting evaluation...")
            val_loss = evaluate(
                device=device,
                valset=valset,
                model=model,
                criterion=criterion,
                writer=writer if writer else SummaryWriter(),  # type: ignore
                epoch=0,  # Not applicable for standalone evaluation
            )
            print("✓ Evaluation completed")
            print(f"Final validation loss: {val_loss:.6f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    finally:
        # Close TensorBoard writer if it was created
        if writer:
            writer.close()  # type: ignore
            print("TensorBoard logs saved")


if __name__ == "__main__":
    main()
