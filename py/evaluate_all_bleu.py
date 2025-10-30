"""
Evaluate all models in a directory by calculating BLEU scores.

This script finds all model files in a directory and calculates their BLEU scores
by translating a test dataset, detokenizing the output, and comparing to reference.

Usage:
    python evaluate_all_bleu.py <models_directory>
    python evaluate_all_bleu.py ../5_checkpoints/
    python evaluate_all_bleu.py ../5_checkpoints/ --pattern "checkpoint_*.pt"
    python evaluate_all_bleu.py ../5_checkpoints/ --output-csv bleu_results.csv
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import torch

from bleu import get_bleu_score
from inference import translate_dataset
from model import Transformer
from tokenization import detokenize_dataset
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
    directory = os.path.normpath(directory)
    search_pattern = os.path.join(directory, pattern)

    model_files = glob.glob(search_pattern)

    if not model_files:
        return []

    # Sort by epoch number extracted from filename
    def extract_epoch_number(filepath: str) -> int:
        filename = os.path.basename(filepath)
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 0

    return sorted(model_files, key=extract_epoch_number)


def get_bleu_for_model(
    model_path: str,
    test_dataset: str,
    reference_file: str,
    tokenizer_json: str,
    device: torch.device,
    beam_size: int = 4,
    output_dir: str = "../tmp",
) -> float:
    """
    Get BLEU score for a single model.

    Args:
        model_path: Path to model weights file
        test_dataset: Path to test dataset (without .bin/.idx suffix)
        reference_file: Path to reference text file
        tokenizer_json: Path to tokenizer JSON file
        device: torch device to use
        beam_size: Beam size for translation
        output_dir: Directory for temporary files

    Returns:
        BLEU score
    """
    model_name = Path(model_path).stem

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate temp file paths
    translated_path = os.path.join(output_dir, f"{model_name}.translated")
    detokenized_path = os.path.join(output_dir, f"{model_name}.translated.txt")

    try:
        # Step 1: Load model and translate
        model = Transformer()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        translate_dataset(device, model, test_dataset, translated_path, beam_size=beam_size)

        # Step 2: Detokenize
        detokenize_dataset(tokenizer_json, translated_path, detokenized_path)

        # Step 3: Calculate BLEU score
        bleu_score = get_bleu_score(reference_file, detokenized_path)

        return bleu_score

    finally:
        # Clean up temporary files
        for ext in [".bin", ".idx"]:
            temp_file = translated_path + ext
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(detokenized_path):
            os.remove(detokenized_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all models in a directory by calculating BLEU scores"
    )
    parser.add_argument(
        "models_directory",
        help="Path to directory containing model weights files (e.g., ../5_checkpoints/)",
    )
    parser.add_argument(
        "--test-dataset",
        default="../4_tokens/newstest2013",
        help="Path to test dataset without suffix (default: ../4_tokens/newstest2013)",
    )
    parser.add_argument(
        "--reference-file",
        default="../1_input/newstest2013.de",
        help="Path to reference text file (default: ../1_input/newstest2013.de)",
    )
    parser.add_argument(
        "--tokenizer",
        default="../3_tokenizer/tokenizer.json",
        help="Path to tokenizer JSON file (default: ../3_tokenizer/tokenizer.json)",
    )
    parser.add_argument(
        "--pattern",
        default="model_*.pt",
        help="Glob pattern to match model files (default: model_*.pt)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for translation (default: 4)",
    )
    parser.add_argument(
        "--output-dir",
        default="../tmp",
        help="Directory for temporary files (default: ../tmp)",
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

    # Check if required files exist
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file not found: {args.reference_file}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
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

    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Beam size: {args.beam_size}")
    print()

    # Store results
    results: List[Tuple[str, float]] = []

    # Evaluate each model
    try:
        for i, model_path in enumerate(model_files, 1):
            model_name = os.path.basename(model_path)
            print(f"[{i}/{len(model_files)}] Evaluating {model_name}")

            try:
                bleu_score = get_bleu_for_model(
                    model_path,
                    args.test_dataset,
                    args.reference_file,
                    args.tokenizer,
                    device,
                    args.beam_size,
                    args.output_dir,
                )

                results.append((model_name, bleu_score))
                print(f"    BLEU: {bleu_score:.2f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append((model_name, float("-inf")))
                continue

            print()

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")

    # Show summary
    print("-" * 50)
    valid_results = [(name, bleu) for name, bleu in results if bleu != float("-inf")]

    if valid_results:
        # Sort by BLEU score (descending)
        valid_results.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Model':<20} {'BLEU Score':<10}")
        print("-" * 30)
        for model_name, bleu_score in valid_results:
            print(f"{model_name:<20} {bleu_score:<10.2f}")

        best_model, best_bleu = valid_results[0]
        print(f"\nBest: {best_model} (BLEU: {best_bleu:.2f})")

        # Save results to CSV if requested
        if args.output_csv:
            try:
                import csv

                with open(args.output_csv, "w", newline="") as csvfile:
                    writer_csv = csv.writer(csvfile)
                    writer_csv.writerow(["Model", "BLEU_Score"])
                    for model_name, bleu_score in valid_results:
                        writer_csv.writerow([model_name, f"{bleu_score:.2f}"])
                print(f"\nResults saved to: {args.output_csv}")
            except Exception as e:
                print(f"Failed to save CSV: {e}")
    else:
        print("No models evaluated successfully")


if __name__ == "__main__":
    main()
