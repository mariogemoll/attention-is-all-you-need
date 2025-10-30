"""
Get BLEU score for a model by translating, detokenizing, and scoring.

This script combines three steps into one:
1. Translate a dataset using the model
2. Detokenize the output
3. Calculate BLEU score against reference

Usage:
    python get_bleu_for_model.py <model_path> <test_dataset> <reference_file> <tokenizer_json>
    python get_bleu_for_model.py ../5_checkpoints/model_0001.pt ../4_tokens/newstest2013 \
        ../1_input/newstest2013.de ../3_tokenizer/tokenizer.json

Optional arguments:
    --beam-size: Beam size for translation (default: 4)
    --keep-temp: Keep temporary files (default: delete them)
    --output-dir: Directory for temporary files (default: ../tmp)
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from bleu import get_bleu_score
from inference import translate_dataset
from model import Transformer
from tokenization import detokenize_dataset
from util import get_device


def get_bleu_for_model(
    model_path: str,
    test_dataset: str,
    reference_file: str,
    tokenizer_json: str,
    beam_size: int = 4,
    keep_temp: bool = False,
    output_dir: str = "../tmp",
) -> float:
    """
    Get BLEU score for a model on a test dataset.

    Args:
        model_path: Path to model weights file
        test_dataset: Path to test dataset (without .bin/.idx suffix)
        reference_file: Path to reference text file
        tokenizer_json: Path to tokenizer JSON file
        beam_size: Beam size for translation
        keep_temp: Whether to keep temporary files
        output_dir: Directory for temporary files

    Returns:
        BLEU score
    """
    device = get_device()
    model_name = Path(model_path).stem

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate temp file paths
    translated_path = os.path.join(output_dir, f"{model_name}.translated")
    detokenized_path = os.path.join(output_dir, f"{model_name}.translated.txt")

    try:
        # Step 1: Load model and translate
        print(f"Loading model: {model_path}")
        model = Transformer()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        print(f"Translating dataset: {test_dataset}")
        translate_dataset(device, model, test_dataset, translated_path, beam_size=beam_size)

        # Step 2: Detokenize
        print("Detokenizing output")
        detokenize_dataset(tokenizer_json, translated_path, detokenized_path)

        # Step 3: Calculate BLEU score
        print("Calculating BLEU score")
        bleu_score = get_bleu_score(reference_file, detokenized_path)

        return bleu_score

    finally:
        # Clean up temporary files unless user wants to keep them
        if not keep_temp:
            for ext in [".bin", ".idx"]:
                temp_file = translated_path + ext
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(detokenized_path):
                os.remove(detokenized_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get BLEU score for a model (translate + detokenize + score)"
    )
    parser.add_argument(
        "model_path",
        help="Path to model weights file (e.g., ../5_checkpoints/model_0001.pt)",
    )
    parser.add_argument(
        "test_dataset",
        help="Path to test dataset without suffix (e.g., ../4_tokens/newstest2013)",
    )
    parser.add_argument(
        "reference_file",
        help="Path to reference text file (e.g., ../1_input/newstest2013.de)",
    )
    parser.add_argument(
        "tokenizer_json",
        help="Path to tokenizer JSON file (e.g., ../3_tokenizer/tokenizer.json)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for translation (default: 4)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary translation files",
    )
    parser.add_argument(
        "--output-dir",
        default="../tmp",
        help="Directory for temporary files (default: ../tmp)",
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file not found: {args.reference_file}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer_json):
        print(f"Error: Tokenizer file not found: {args.tokenizer_json}")
        sys.exit(1)

    device = get_device()
    print(f"Using device: {device}")
    print()

    bleu_score = get_bleu_for_model(
        args.model_path,
        args.test_dataset,
        args.reference_file,
        args.tokenizer_json,
        args.beam_size,
        args.keep_temp,
        args.output_dir,
    )

    print()
    print(f"BLEU score: {bleu_score:.2f}")


if __name__ == "__main__":
    main()
