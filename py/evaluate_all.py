#!/usr/bin/env python3
"""
Evaluate all models in a directory on validation dataset.

Usage:
    python evaluate_all.py <models_directory>
    python evaluate_all.py ../5_checkpoints/
    python evaluate_all.py ../5_checkpoints/ --pattern "checkpoint_*.pt"
    python evaluate_all.py ../5_checkpoints/ --output-csv results.csv
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from buckets import open_buckets
from model import Transformer
from params import pad
from training import evaluate
from util import get_device


def find_model_files(directory: str, pattern: str = "model_*.pt") -> List[str]:
    """
    Find all model files in the given directory matching the pattern.

    Args:
        directory: Path to the directory containing model files
        pattern: Glob pattern to match model files (default: "model_*.pt")

    Returns:
        List of model file paths, sorted by epoch number
    """
    # Ensure directory path has proper separator
    directory = os.path.normpath(directory)
    search_pattern = os.path.join(directory, pattern)

    model_files = glob.glob(search_pattern)

    if not model_files:
        return []

    # Sort by epoch number extracted from filename
    def extract_epoch_number(filepath: str) -> int:
        filename = os.path.basename(filepath)
        # Extract numbers from filename (e.g., "model_0005.pt" -> 5)
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 0

    return sorted(model_files, key=extract_epoch_number)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models in a directory on validation dataset"
    )
    parser.add_argument(
        "models_directory",
        help="Path to directory containing model weights files (e.g., ../5_checkpoints/)",
    )
    parser.add_argument(
        "--dataset",
        default="../4_tokens/newstest2013",
        help="Path to validation dataset (default: ../4_tokens/newstest2013)",
    )
    parser.add_argument(
        "--pattern",
        default="model_*.pt",
        help="Glob pattern to match model files (default: model_*.pt)",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to save results as CSV file",
    )

    args = parser.parse_args()

    # Check if directory exists
    if not os.path.isdir(args.models_directory):
        print(f"Error: Directory not found: {args.models_directory}")
        sys.exit(1)

    # Find all model files
    model_files = find_model_files(args.models_directory, args.pattern)

    if not model_files:
        print(f"No model files found in {args.models_directory} matching pattern '{args.pattern}'")
        sys.exit(1)

    print(f"Found {len(model_files)} model files to evaluate:")
    for model_file in model_files:
        print(f"  - {os.path.basename(model_file)}")
    print()

    # Initialize device and TensorBoard writer
    device = get_device()
    writer = SummaryWriter()  # type: ignore[no-untyped-call]
    print(f"Using device: {device}")

    # Store results
    results: List[Tuple[str, float]] = []

    # Initialize loss criterion
    criterion = CrossEntropyLoss(ignore_index=pad)

    # Evaluate each model
    try:
        for i, model_path in enumerate(model_files, 1):
            model_name = os.path.basename(model_path)
            print(f"[{i}/{len(model_files)}] {model_name}")

            try:
                # Extract epoch number from filename
                match = re.search(r"(\d+)", model_name)
                if not match:
                    raise ValueError(f"Could not extract epoch number from filename: {model_name}")
                epoch = int(match.group(1))

                # Load model
                model = Transformer().to(device)
                model_state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(model_state_dict)

                # Evaluate
                with open_buckets(args.dataset) as valset:
                    val_loss = evaluate(
                        device=device,
                        valset=valset,
                        model=model,
                        criterion=criterion,
                        writer=writer,
                        epoch=epoch,
                    )

                results.append((model_name, val_loss))
                print(f"    Loss: {val_loss:.6f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append((model_name, float("inf")))
                continue

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")

    finally:
        # Close TensorBoard writer
        writer.close()  # type: ignore[no-untyped-call]
        print("TensorBoard logs saved")

    # Show summary
    print("-" * 50)
    valid_results = [(name, loss) for name, loss in results if loss != float("inf")]

    if valid_results:
        # Sort by loss and show all results
        valid_results.sort(key=lambda x: x[1])
        print(f"{'Model':<20} {'Loss':<10}")
        print("-" * 30)
        for model_name, val_loss in valid_results:
            print(f"{model_name:<20} {val_loss:<10.6f}")

        best_model, best_loss = valid_results[0]
        print(f"\nBest: {best_model} (loss: {best_loss:.6f})")

        # Save results to CSV if requested
        if args.output_csv:
            try:
                import csv

                with open(args.output_csv, "w", newline="") as csvfile:
                    writer_csv = csv.writer(csvfile)
                    writer_csv.writerow(["Model", "Validation_Loss"])
                    for model_name, val_loss in valid_results:
                        writer_csv.writerow([model_name, f"{val_loss:.6f}"])
                print(f"Results saved to: {args.output_csv}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")
    else:
        print("No models evaluated successfully")


if __name__ == "__main__":
    main()
