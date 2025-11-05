#!/usr/bin/env python3
"""
Compute BLEU scores for multiple model translations produced by
`translate_dataset_multimodel.py`.

The script expects a root translations directory (for one dataset) where each
model has a subdirectory (named by model file) containing the translated files
created by the translation script. It looks for `translation.txt` inside each
model's subdirectory and computes BLEU against the provided reference file.

Uses multiprocessing with one process per model, respecting max_parallelism from params.py.

Usage:
    python get_bleu_score_multimodel.py <reference_file> <translations_root>
        [--output-csv results.csv]

Example:
    python get_bleu_score_multimodel.py newstest2013.ref.txt ../6_translations/newstest2013
        --output-csv newstest2013_bleu.csv
"""

import argparse
import concurrent.futures
import csv
import os
import sys
from typing import List, Optional, Tuple

from bleu import get_bleu_score
from params import max_parallelism


def find_model_translation_file(model_dir: str) -> str:
    """Return path to the translation text file inside a model directory.

    The translation script writes the detokenized file as `translation.txt`
    (i.e., <model_dir>/translation.txt). If not found, this function raises
    FileNotFoundError.
    """
    candidate = os.path.join(model_dir, "translation.txt")
    if os.path.exists(candidate):
        return candidate
    # Backwards-compat: allow plain .txt files in the directory (pick the first)
    for entry in os.listdir(model_dir):
        if entry.endswith(".txt"):
            return os.path.join(model_dir, entry)
    raise FileNotFoundError(f"No translation text file found in {model_dir}")


def collect_model_dirs(root: str) -> List[str]:
    """Collect immediate subdirectories under root, sorted alphabetically."""
    try:
        entries = os.listdir(root)
    except FileNotFoundError:
        raise
    dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
    dirs.sort()
    return dirs


def evaluate_single_model(
    model_name: str, translations_root: str, reference_file: str
) -> Tuple[str, Optional[float], Optional[str]]:
    """Evaluate BLEU for a single model.

    Returns (model_name, bleu_score, error_message).
    If successful, error_message is None.
    If failed, bleu_score is None and error_message contains the error.
    """
    model_dir = os.path.join(translations_root, model_name)
    try:
        translation_txt = find_model_translation_file(model_dir)
    except FileNotFoundError:
        return (model_name, None, "no translation text file")

    try:
        score = get_bleu_score(reference_file, translation_txt)
        return (model_name, score, None)
    except AssertionError as e:
        return (model_name, None, str(e))
    except Exception as e:
        return (model_name, None, f"Error: {e}")


def evaluate_all(translations_root: str, reference_file: str) -> List[Tuple[str, float]]:
    """Evaluate BLEU for all model subdirs under translations_root using multiprocessing.

    Returns a list of (model_name, bleu_score) for successful evaluations only.
    """
    models = collect_model_dirs(translations_root)

    if not models:
        print(f"No model subdirectories found in {translations_root}")
        return []

    print(f"Found {len(models)} model(s) in {translations_root}")

    # Determine number of workers
    max_workers = min(max_parallelism, len(models))
    print(f"Using {max_workers} workers for BLEU evaluation")

    results: List[Tuple[str, float]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_model = {
            executor.submit(
                evaluate_single_model, model_name, translations_root, reference_file
            ): model_name
            for model_name in models
        }

        # Process results
        for future in concurrent.futures.as_completed(future_to_model):
            model_name, score, error = future.result()

            if score is not None:
                print(f"{model_name}: BLEU={score:.2f}")
                results.append((model_name, score))
            else:
                print(f"Skipping {model_name}: {error}")

    return results


def write_csv(results: List[Tuple[str, float]], output_csv: str) -> None:
    """Write results to CSV with header (Model, BLEU) sorted alphabetically by model."""
    results_sorted = sorted(results, key=lambda x: x[0])
    try:
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Model", "BLEU"])
            for model_name, score in results_sorted:
                writer.writerow([model_name, f"{score:.6f}"])
        print(f"Results written to {output_csv}")
    except Exception as e:
        print(f"Failed to write CSV {output_csv}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute BLEU scores for all model translations in a directory"
    )
    parser.add_argument("reference_file", help="Reference file to compute BLEU against")
    parser.add_argument(
        "translations_root",
        help=(
            "Root directory containing model subdirectories (e.g., ../6_translations/newstest2013)"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="bleu_results.csv",
        help="CSV output path (default: bleu_results.csv)",
    )

    args = parser.parse_args()

    translations_root = args.translations_root
    reference_file = args.reference_file
    output_csv = args.output_csv

    if not os.path.isdir(translations_root):
        print(f"Error: translations root not found: {translations_root}")
        sys.exit(1)
    if not os.path.exists(reference_file):
        print(f"Error: reference file not found: {reference_file}")
        sys.exit(1)

    results = evaluate_all(translations_root, reference_file)

    if results:
        write_csv(results, output_csv)
    else:
        print("No BLEU scores to write.")


if __name__ == "__main__":
    main()
