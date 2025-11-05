#!/usr/bin/env python3
"""
Alternative multiprocessing evaluation using subprocess for better isolation.

Usage:
    python evaluate_all_subprocess.py <models_directory>
    python evaluate_all_subprocess.py ../5_checkpoints/ --max-workers 4
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import torch

from evaluate_all import find_model_files


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def evaluate_model_subprocess(
    model_path: str, dataset_path: str, gpu_id: Optional[int] = None
) -> Tuple[str, float, bool]:
    """
    Evaluate a model using subprocess for better isolation.

    Args:
        model_path: Path to model file
        dataset_path: Path to validation dataset
        gpu_id: GPU device ID to use

    Returns:
        Tuple of (model_name, validation_loss, success)
    """
    model_name = os.path.basename(model_path)

    # Build command
    cmd = [
        sys.executable,
        "evaluate_model_process.py",
        model_path,
        "--dataset",
        dataset_path,
        "--quiet",
    ]

    if gpu_id is not None:
        cmd.extend(["--gpu", str(gpu_id)])

    try:
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(__file__) or ".",
        )

        if result.returncode == 0:
            # Parse validation loss from stdout
            try:
                val_loss = float(result.stdout.strip().split("\n")[-1])
                return (model_name, val_loss, True)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse output for {model_name}: {result.stdout}")
                return (model_name, float("inf"), False)
        else:
            print(f"Error evaluating {model_name}: {result.stderr}")
            return (model_name, float("inf"), False)

    except subprocess.TimeoutExpired:
        print(f"Timeout evaluating {model_name}")
        return (model_name, float("inf"), False)
    except Exception as e:
        print(f"Exception evaluating {model_name}: {e}")
        return (model_name, float("inf"), False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models using subprocess-based multiprocessing"
    )
    parser.add_argument("models_directory", help="Path to directory containing model weights files")
    parser.add_argument(
        "--dataset", default="../4_tokens/newstest2013", help="Path to validation dataset"
    )
    parser.add_argument("--pattern", default="model_*.pt", help="Glob pattern to match model files")
    parser.add_argument("--output-csv", help="Optional path to save results as CSV file")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: number of GPUs or CPU count)",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only evaluation")

    args = parser.parse_args()

    # Check directory
    if not os.path.isdir(args.models_directory):
        print(f"Error: Directory not found: {args.models_directory}")
        sys.exit(1)

    # Find model files
    model_files = find_model_files(args.models_directory, args.pattern)
    if not model_files:
        print(f"No model files found in {args.models_directory}")
        sys.exit(1)

    print(f"Found {len(model_files)} model files to evaluate:")
    for model_file in model_files[:5]:  # Show first 5
        print(f"  - {os.path.basename(model_file)}")
    if len(model_files) > 5:
        print(f"  ... and {len(model_files) - 5} more")
    print()

    # Determine workers and GPUs
    if args.cpu_only:
        available_gpus = []
    else:
        available_gpus = get_available_gpus()

    if args.max_workers:
        max_workers = args.max_workers
    else:
        if available_gpus:
            max_workers = len(available_gpus)
        else:
            max_workers = min(os.cpu_count() or 1, len(model_files))

    print(f"Available GPUs: {available_gpus if available_gpus else 'None'}")
    print(f"Using {max_workers} workers")
    print()

    # Create tasks
    tasks = []
    for i, model_path in enumerate(model_files):
        gpu_id = None
        if available_gpus and not args.cpu_only:
            gpu_id = available_gpus[i % len(available_gpus)]

        tasks.append((model_path, args.dataset, gpu_id))

    # Execute with ThreadPoolExecutor (subprocess handles the multiprocessing)
    results = []
    start_time = time.time()

    print("Starting evaluation...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(evaluate_model_subprocess, model_path, dataset, gpu_id): (
                model_path,
                gpu_id,
            )
            for model_path, dataset, gpu_id in tasks
        }

        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_task):
            model_path, gpu_id = future_to_task[future]
            try:
                model_name, val_loss, success = future.result()
                results.append((model_name, val_loss, success))

                status = f"loss: {val_loss:.6f}" if success else "FAILED"
                gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else " (CPU)"
                print(f"{model_name}: {status}{gpu_info}")

            except Exception as exc:
                model_name = os.path.basename(model_path)
                results.append((model_name, float("inf"), False))
                print(f"{model_name}: Exception - {exc}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Show summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    valid_results = [(name, loss) for name, loss, success in results if success]
    failed_results = [(name, loss) for name, loss, success in results if not success]

    if valid_results:
        # Sort by loss
        valid_results.sort(key=lambda x: x[1])
        print(f"{'Model':<25} {'Loss':<12}")
        print("-" * 40)
        for model_name, val_loss in valid_results:
            print(f"{model_name:<25} {val_loss:<12.6f}")

        best_model, best_loss = valid_results[0]
        print(f"\nBest: {best_model} (loss: {best_loss:.6f})")

    if failed_results:
        print(f"\nFailed evaluations: {len(failed_results)}")
        for name, _ in failed_results:
            print(f"   - {name}")

    # Save CSV
    if args.output_csv:
        try:
            import csv

            with open(args.output_csv, "w", newline="") as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow(["Model", "Validation_Loss", "Status"])
                for model_name, val_loss, success in results:
                    status = "Success" if success else "Failed"
                    loss_str = f"{val_loss:.6f}" if success else "N/A"
                    writer_csv.writerow([model_name, loss_str, status])
            print(f"Results saved to: {args.output_csv}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    success_count = sum(1 for _, _, success in results if success)
    print(f"\nSuccessfully evaluated: {success_count}/{len(results)} models")


if __name__ == "__main__":
    main()
