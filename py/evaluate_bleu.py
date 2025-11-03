#!/usr/bin/env python3
"""
Standalone evaluation script for calculating BLEU score on a single model.

This script loads a trained model, translates a test dataset, detokenizes the output,
and calculates the BLEU score by comparing to a reference file.

Usage:
    python evaluate_bleu.py <model_weights_file>
    python evaluate_bleu.py model_0005.pt
    python evaluate_bleu.py model_0005.pt --test-dataset ../4_tokens/newstest2014.en-de
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a single trained model by calculating BLEU score"
    )
    parser.add_argument(
        "model_weights_file",
        help="Path to the model weights file (e.g., model_0005.pt)",
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
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary translation files (default: remove them)",
    )

    args = parser.parse_args()

    # Check if model weights file exists
    if not os.path.exists(args.model_weights_file):
        print(f"Error: Model weights file not found: {args.model_weights_file}")
        sys.exit(1)

    # Check if required files exist
    if not os.path.exists(args.reference_file):
        print(f"Error: Reference file not found: {args.reference_file}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file not found: {args.tokenizer}")
        sys.exit(1)

    torch.set_float32_matmul_precision("high")
    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {os.path.basename(args.model_weights_file)}")
    print(f"Test dataset: {args.test_dataset}")
    print(f"Reference: {args.reference_file}")
    print(f"Beam size: {args.beam_size}")
    print()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate temp file paths
    model_name = Path(args.model_weights_file).stem
    translated_path = os.path.join(args.output_dir, f"{model_name}.translated")
    detokenized_path = os.path.join(args.output_dir, f"{model_name}.translated.txt")

    try:
        # Step 1: Load model
        print("Loading model...")
        model = Transformer()
        try:
            model_state_dict = torch.load(args.model_weights_file, map_location=device)
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.compile()  # type: ignore[no-untyped-call]
            model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)

        # Step 2: Translate dataset
        print("\nTranslating test dataset...")
        try:
            translate_dataset(
                device, model, args.test_dataset, translated_path, beam_size=args.beam_size
            )
            print("✓ Translation completed")
        except Exception as e:
            print(f"Error during translation: {e}")
            sys.exit(1)

        # Step 3: Detokenize
        print("\nDetokenizing output...")
        try:
            detokenize_dataset(args.tokenizer, translated_path, detokenized_path)
            print("✓ Detokenization completed")
        except Exception as e:
            print(f"Error during detokenization: {e}")
            sys.exit(1)

        # Step 4: Calculate BLEU score
        print("\nCalculating BLEU score...")
        try:
            bleu_score = get_bleu_score(args.reference_file, detokenized_path)
            print("✓ BLEU calculation completed")
            print()
            print("=" * 50)
            print(f"BLEU Score: {bleu_score:.2f}")
            print("=" * 50)
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            sys.exit(1)

    finally:
        # Clean up temporary files unless requested to keep them
        if not args.keep_temp_files:
            print("\nCleaning up temporary files...")
            for ext in [".bin", ".idx"]:
                temp_file = translated_path + ext
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"  Removed: {temp_file}")
            if os.path.exists(detokenized_path):
                os.remove(detokenized_path)
                print(f"  Removed: {detokenized_path}")
        else:
            print(f"\nTranslated output saved to: {detokenized_path}")


if __name__ == "__main__":
    main()
