#!/usr/bin/env python3
import argparse
import sys

import tokenizers  # type: ignore
from tabulate import tabulate

from tokenization import decode, sample_from_dataset


def display_sample(
    tokenizer: tokenizers.Tokenizer,
    sample_index: int,
    corpus_id: int,
    original_line_number: int,
    src_tokens: list[int],
    tgt_tokens: list[int],
) -> None:
    """Display a sampled entry in a readable format."""
    print(f"Sample {sample_index}, corpus ID: {corpus_id}, original line: {original_line_number}")
    print()

    chunk_len = 20

    src_rows = [src_tokens[i : i + chunk_len] for i in range(0, len(src_tokens), chunk_len)]
    for tokens in src_rows:
        print(tabulate([[decode(tokenizer, token) for token in tokens], tokens]))

    print()

    tgt_rows = [tgt_tokens[i : i + chunk_len] for i in range(0, len(tgt_tokens), chunk_len)]
    for tokens in tgt_rows:
        print(tabulate([[decode(tokenizer, token) for token in tokens], tokens]))

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample entries from a binary dataset and display them"
    )
    parser.add_argument("tokenizer_path", help="Path to the tokenizer JSON file")
    parser.add_argument("dataset_path", help="Path to the dataset file (without suffix)")
    parser.add_argument("num_samples", type=int, help="Number of samples to display")

    args = parser.parse_args()

    # Load tokenizer
    try:
        tokenizer = tokenizers.Tokenizer.from_file(args.tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

    # Sample from dataset
    try:
        samples = sample_from_dataset(args.dataset_path, args.num_samples)
    except Exception as e:
        print(f"Error sampling dataset: {e}")
        sys.exit(1)

    # Display samples
    for i, (corpus_id, original_line_number, src_tokens, tgt_tokens) in enumerate(samples, 1):
        display_sample(tokenizer, i, corpus_id, original_line_number, src_tokens, tgt_tokens)


if __name__ == "__main__":
    main()
