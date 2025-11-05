#!/usr/bin/env python3
"""
Translate dataset using multiple models with multiprocessing.

This script translates a bucketed dataset using multiple/all models found in a directory.
Each model's output is stored in a separate subdirectory with the model name.
The script outputs both tokenized results and optionally detokenized text files.

The input dataset should be in bucketed format with files:
- <dataset>.bidx (bucket index)
- <dataset>.src.bin/.src.idx (source data)
- <dataset>.tgt.bin/.tgt.idx (target data)
- <dataset>.meta (metadata)

Usage:
    python translate_dataset_multimodel.py <input_dataset> <output_directory>
    python translate_dataset_multimodel.py ../4_tokens/newstest2013 ../6_translations/newstest2013
    python translate_dataset_multimodel.py ../4_tokens/newstest2014 ../6_translations/newstest2014
    python translate_dataset_multimodel.py ../4_tokens/newstest2013 ../6_translations/newstest2013
        --beam-size 5 --max-workers 4
    python translate_dataset_multimodel.py ../4_tokens/newstest2013 ../6_translations/newstest2013
        --models-directory ../my_checkpoints/
    python translate_dataset_multimodel.py ../4_tokens/newstest2013 ../6_translations/newstest2013
        --no-detokenize
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
from tokenization import detokenize_dataset


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"


def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def translate_and_detokenize_model(
    model_path: str,
    input_dataset: str,
    output_dir: str,
    tokenizer_path: str,
    beam_size: int = 4,
    gpu_id: Optional[int] = None,
    skip_detokenize: bool = False,
) -> Tuple[str, bool, str]:
    """
    Translate dataset using a model and optionally detokenize the result.

    Args:
        model_path: Path to model file
        input_dataset: Path to input dataset
        output_dir: Base output directory
        tokenizer_path: Path to tokenizer JSON file
        beam_size: Beam size for translation
        gpu_id: GPU device ID to use
        skip_detokenize: Whether to skip detokenization

    Returns:
        Tuple of (model_name, success, output_path)
    """
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Create model-specific output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Output path for the translated dataset
    output_dataset = os.path.join(model_output_dir, "translation")
    output_txt_path = output_dataset + ".txt"

    # Build translation command
    cmd = [
        sys.executable,
        "translate_dataset.py",
        model_path,
        input_dataset,
        output_dataset,
        str(beam_size),
    ]

    # Add --show-progress flag for GPU 0 only
    if gpu_id == 0:
        cmd.append("--show-progress")

    # Set GPU environment variable if specified
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Step 1: Translation
        if gpu_id == 0:
            # For GPU 0, don't capture output so tqdm can display
            result = subprocess.run(
                cmd,
                text=True,  # Add text=True for consistent typing
                timeout=1800,  # 30 minute timeout
                cwd=os.path.dirname(__file__) or ".",
                env=env,
            )
            translation_success = result.returncode == 0
        else:
            # For other GPUs, capture output to keep it quiet
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                cwd=os.path.dirname(__file__) or ".",
                env=env,
            )
            translation_success = result.returncode == 0

        if not translation_success:
            if gpu_id != 0 and result.stderr:
                print(f"Error translating with {model_name}: {result.stderr}")
            return (model_name, False, output_dataset)

        # Step 2: Detokenization (if not skipped)
        if not skip_detokenize:
            try:
                detokenize_dataset(tokenizer_path, output_dataset, output_txt_path)
                if gpu_id == 0:
                    print(f"Detokenized {model_name} -> {os.path.basename(output_txt_path)}")
            except Exception as e:
                print(f"Warning: Detokenization failed for {model_name}: {e}")
                # Don't fail the whole process if detokenization fails

        return (model_name, True, output_dataset)

    except subprocess.TimeoutExpired:
        print(f"Timeout translating with {model_name}")
        return (model_name, False, output_dataset)
    except Exception as e:
        print(f"Exception translating with {model_name}: {e}")
        return (model_name, False, output_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate dataset using multiple models with multiprocessing"
    )
    parser.add_argument(
        "input_dataset",
        help=(
            "Path to input bucketed dataset (without file extensions, e.g., "
            "'../4_tokens/newstest2013')"
        ),
    )
    parser.add_argument(
        "output_directory",
        help=(
            "Base output directory (model subdirs will be created here, e.g., "
            "'../6_translations/newstest2013')"
        ),
    )
    parser.add_argument(
        "--models-directory",
        default="../5_checkpoints",
        help="Path to directory containing model weights files (default: ../5_checkpoints)",
    )
    parser.add_argument(
        "--pattern",
        default="model_*.pt",
        help="Glob pattern to match model files (default: model_*.pt)",
    )
    parser.add_argument(
        "--beam-size", type=int, default=4, help="Beam size for translation (default: 4)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: number of GPUs or CPU count)",
    )
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU-only translation")
    parser.add_argument(
        "--no-detokenize",
        action="store_true",
        help="Skip detokenization step (only output binary tokens)",
    )
    parser.add_argument(
        "--tokenizer",
        default="../3_tokenizer/tokenizer.json",
        help="Path to tokenizer JSON file (default: ../3_tokenizer/tokenizer.json)",
    )

    args = parser.parse_args()

    # Check inputs
    if not os.path.isdir(args.models_directory):
        print(f"Error: Models directory not found: {args.models_directory}")
        sys.exit(1)

    # Check for bucketed dataset files (.bidx, .src.bin/.src.idx, .tgt.bin/.tgt.idx, .meta)
    required_files = [
        args.input_dataset + ".bidx",
        args.input_dataset + ".src.bin",
        args.input_dataset + ".src.idx",
        args.input_dataset + ".tgt.bin",
        args.input_dataset + ".tgt.idx",
        args.input_dataset + ".meta",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Input dataset files not found:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    if not args.no_detokenize and not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_directory, exist_ok=True)

    # Find model files
    model_files = find_model_files(args.models_directory, args.pattern)
    if not model_files:
        print(f"No model files found in {args.models_directory}")
        sys.exit(1)

    print(f"Found {len(model_files)} model files to use for translation:")
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
    print(f"Beam size: {args.beam_size}")
    print(f"Detokenization: {'Enabled' if not args.no_detokenize else 'Disabled'}")
    print()

    # Create translation tasks
    translation_tasks = []
    for i, model_path in enumerate(model_files):
        gpu_id = None
        if available_gpus and not args.cpu_only:
            gpu_id = available_gpus[i % len(available_gpus)]

        translation_tasks.append(
            (
                model_path,
                args.input_dataset,
                args.output_directory,
                args.beam_size,
                gpu_id,
            )
        )

    # Execute translation with ThreadPoolExecutor
    translation_results = []
    start_time = time.time()
    model_times = []  # Track completion times for ETA calculation

    print("Starting translations...")
    print()  # Add spacing for cleaner output

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks with start times
        future_to_task = {}
        task_start_times = {}

        for model_path, input_dataset, output_dir, beam_size, gpu_id in translation_tasks:
            task_start = time.time()
            future = executor.submit(
                translate_and_detokenize_model,
                model_path,
                input_dataset,
                output_dir,
                args.tokenizer,
                beam_size,
                gpu_id,
                args.no_detokenize,
            )
            future_to_task[future] = (model_path, gpu_id)
            task_start_times[future] = task_start

        # Process completed futures without top-level progress bar
        completed_count = 0

        for future in concurrent.futures.as_completed(future_to_task):
            model_path, gpu_id = future_to_task[future]
            task_end = time.time()
            model_duration = task_end - task_start_times[future]

            try:
                model_name, success, output_dataset = future.result()
                translation_results.append((model_name, success, output_dataset))
                completed_count += 1

                if success:
                    model_times.append(model_duration)

                status = "SUCCESS" if success else "FAILED"
                gpu_info = f" (GPU {gpu_id})" if gpu_id is not None else " (CPU)"

                # Calculate timing information
                elapsed_total = task_end - start_time
                elapsed_str = format_duration(elapsed_total)
                model_time_str = format_duration(model_duration)

                # Calculate ETA and statistics
                remaining_models = len(translation_tasks) - completed_count
                if model_times and remaining_models > 0:
                    avg_model_time = sum(model_times) / len(model_times)
                    # Account for parallel processing
                    eta_seconds = (remaining_models * avg_model_time) / max_workers
                    eta_str = format_duration(eta_seconds)
                    avg_str = format_duration(avg_model_time)

                    # Estimate total time
                    estimated_total = (len(translation_tasks) * avg_model_time) / max_workers
                    total_str = format_duration(estimated_total)
                else:
                    eta_str = "calculating..."
                    avg_str = "calculating..."
                    total_str = "calculating..."

                # Format model times for display
                if model_times:
                    times_display = " | ".join([format_duration(t) for t in model_times])
                else:
                    times_display = "none yet"

                # Print completion status and timing info
                print(f"\n{model_name}: {status}{gpu_info} ({model_time_str})")
                print(f"Progress: {completed_count}/{len(translation_tasks)} models completed")
                print(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                print(f"Model times: {times_display}")
                print(f"Average: {avg_str} | Estimated total: {total_str}")
                print()  # Add spacing for next model

            except Exception as exc:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                translation_results.append((model_name, False, ""))
                completed_count += 1

                elapsed_total = task_end - start_time
                elapsed_str = format_duration(elapsed_total)
                model_time_str = format_duration(model_duration)

                # Calculate statistics (same as success case)
                remaining_models = len(translation_tasks) - completed_count
                if model_times and remaining_models > 0:
                    avg_model_time = sum(model_times) / len(model_times)
                    eta_seconds = (remaining_models * avg_model_time) / max_workers
                    eta_str = format_duration(eta_seconds)
                    avg_str = format_duration(avg_model_time)
                    estimated_total = (len(translation_tasks) * avg_model_time) / max_workers
                    total_str = format_duration(estimated_total)
                else:
                    eta_str = "calculating..."
                    avg_str = "calculating..."
                    total_str = "calculating..."

                # Format model times for display
                if model_times:
                    times_display = " | ".join([format_duration(t) for t in model_times])
                else:
                    times_display = "none yet"

                print(f"\n{model_name}: Exception - {exc} ({model_time_str})")
                print(f"Progress: {completed_count}/{len(translation_tasks)} models completed")
                print(f"Elapsed: {elapsed_str} | ETA: {eta_str}")
                print(f"Model times: {times_display}")
                print(f"Average: {avg_str} | Estimated total: {total_str}")
                print()

    translation_elapsed = time.time() - start_time
    print(f"\nAll translations completed in {translation_elapsed:.1f}s")

    # Show final summary
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRANSLATION SUMMARY")
    print("=" * 60)

    successful_translations = sum(1 for _, success, _ in translation_results if success)
    failed_translations = len(translation_results) - successful_translations

    print(f"Total models: {len(translation_results)}")
    print(f"Successful translations: {successful_translations}")
    print(f"Failed translations: {failed_translations}")
    print(f"Total time: {total_elapsed:.1f}s")

    if failed_translations > 0:
        print("\nFailed models:")
        for model_name, success, _ in translation_results:
            if not success:
                print(f"   - {model_name}")

    print(f"\nOutput directory: {args.output_directory}")
    print("Each model's output is in its own subdirectory:")
    print("  - translation.bin/.idx (tokenized)")
    if not args.no_detokenize:
        print("  - translation.txt (detokenized text, created during translation)")


if __name__ == "__main__":
    main()
