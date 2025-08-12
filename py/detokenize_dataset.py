#!/usr/bin/env python3
"""
Detokenize a binary dataset back to readable text files.

This script reads a binary dataset produced by translate_dataset.py
and converts the tokenized sequences back to human-readable text.

The format is simple:
- .idx file: 5 bytes per entry (4-byte offset + 1-byte length)
- .bin file: 16-bit little-endian integers (tokens)

Usage:
    python detokenize_dataset.py <tokenizer_json> <input_dataset> <output_file>

Example:
    python detokenize_dataset.py tokenizer.json val val.out.txt
"""

import argparse

from tokenization import detokenize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Detokenize binary dataset to readable text file")
    parser.add_argument(
        "tokenizer_json",
        help="Path to the tokenizer JSON file (e.g., 3_tokenizer/tokenizer.json)",
    )
    parser.add_argument(
        "input_dataset",
        help="Path to input binary dataset (without .bin/.idx suffix)",
    )
    parser.add_argument(
        "output_file",
        help="Path to output text file",
    )

    args = parser.parse_args()

    detokenize_dataset(
        args.tokenizer_json,
        args.input_dataset,
        args.output_file,
    )


if __name__ == "__main__":
    main()
