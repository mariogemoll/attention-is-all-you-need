#!/usr/bin/env python3
"""
Standalone model evaluation process for multiprocessing.
This script evaluates a single model and outputs the result.

Usage:
    python evaluate_model_process.py <model_path> [--dataset <dataset_path>] [--gpu <gpu_id>]
"""

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any, Dict, Optional

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from buckets import open_buckets
from evaluation import evaluate
from model import Transformer
from params import pad


def evaluate_model_process(
    model_path: str,
    dataset_path: str = "../4_tokens/newstest2013",
    gpu_id: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model in a separate process.

    Args:
        model_path: Path to the model file
        dataset_path: Path to validation dataset
        gpu_id: GPU device ID (None for auto-detect)
        output_file: Optional file to write JSON results

    Returns:
        Dictionary with evaluation results
    """
    # Set up device
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = os.path.basename(model_path)

    result = {
        "model_path": model_path,
        "model_name": model_name,
        "dataset_path": dataset_path,
        "device": str(device),
        "success": False,
        "validation_loss": None,
        "error": None,
        "epoch": None,
    }

    try:
        # Extract epoch number from filename
        match = re.search(r"(\d+)", model_name)
        if not match:
            raise ValueError(f"Could not extract epoch number from filename: {model_name}")
        epoch = int(match.group(1))
        result["epoch"] = epoch

        # Load model
        model = Transformer().to(device)
        model_state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state_dict)

        # Initialize loss criterion and tensorboard writer
        criterion = CrossEntropyLoss(ignore_index=pad)
        log_dir = f"runs/eval_{model_name}_{os.getpid()}"
        writer = SummaryWriter(log_dir=log_dir)  # type: ignore[no-untyped-call]

        # Evaluate
        with open_buckets(dataset_path) as valset:
            val_loss = evaluate(
                device=device,
                valset=valset,
                model=model,
                criterion=criterion,
                writer=writer,
                epoch=epoch,
            )

        writer.close()  # type: ignore[no-untyped-call]

        result["validation_loss"] = val_loss
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    # Write results to file if specified
    if output_file:
        try:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            result["output_file_error"] = str(e)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a single model in a separate process")
    parser.add_argument("model_path", help="Path to the model weights file")
    parser.add_argument(
        "--dataset",
        default="../4_tokens/newstest2013",
        help="Path to validation dataset (default: ../4_tokens/newstest2013)",
    )
    parser.add_argument("--gpu", type=int, help="GPU device ID to use (optional)")
    parser.add_argument("--output", help="JSON file to write results (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    if not args.quiet:
        print(f"Evaluating: {os.path.basename(args.model_path)}")
        if args.gpu is not None:
            print(f"Using GPU: {args.gpu}")

    # Run evaluation
    result = evaluate_model_process(
        model_path=args.model_path,
        dataset_path=args.dataset,
        gpu_id=args.gpu,
        output_file=args.output,
    )

    # Print results
    if result["success"]:
        if not args.quiet:
            print(f"✓ Success: Loss = {result['validation_loss']:.6f}")
        # Always output the loss for easy parsing
        print(result["validation_loss"])
    else:
        if not args.quiet:
            print(f"✗ Failed: {result['error']}")
        # Output error code
        sys.exit(1)


if __name__ == "__main__":
    main()
