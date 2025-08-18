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
    python detokenize_dataset.py 3_tokenizer/tokenizer.json 4_tokens/translated_val output/translations.txt
"""

import argparse
import os
import struct
import sys

import tokenizers  # type: ignore
import torch
from tqdm import tqdm


def detokenize_sequence(tokenizer: tokenizers.Tokenizer, tokens: list[int]) -> str:
    """
    Convert a list of tokens back to text.

    Args:
        tokenizer: The tokenizer to use for decoding
        tokens: List of token IDs

    Returns:
        Detokenized text string
    """
    if not tokens:
        return ""

    # Remove special tokens and decode the rest
    text_tokens = []
    for token in tokens:
        if token == 0:  # PAD
            continue
        elif token == 1:  # SOS
            continue
        elif token == 2:  # EOS
            break
        else:
            text_tokens.append(token)

    if not text_tokens:
        return ""

    return str(tokenizer.decode(text_tokens))


def read_entry(data_file, index_file, entry_idx: int) -> list[int]:
    """
    Read a single entry from the translate_dataset.py output format.
    
    Args:
        data_file: Open binary data file
        index_file: Open binary index file
        entry_idx: Entry index to read
        
    Returns:
        List of token IDs
    """
    # Read index entry (5 bytes: 4-byte offset + 1-byte length)
    idx_file_pos = entry_idx * 5
    index_file.seek(idx_file_pos)
    index_data = index_file.read(5)
    
    if len(index_data) != 5:
        return []
        
    # Unpack offset and length
    offset = struct.unpack("<I", index_data[:4])[0]  # Little-endian 32-bit
    length = struct.unpack("<B", index_data[4:5])[0]  # Single byte
    
    if length == 0:
        return []
    
    # Read tokens from data file
    data_file.seek(offset)
    token_bytes = data_file.read(length * 2)  # 2 bytes per 16-bit token
    
    if len(token_bytes) != length * 2:
        return []
        
    # Convert bytes to tokens (16-bit little-endian integers)
    tokens_tensor = torch.frombuffer(token_bytes, dtype=torch.int16)
    return tokens_tensor.tolist()


def detokenize_dataset(
    tokenizer_path: str,
    input_dataset_path: str,
    output_file_path: str,
) -> None:
    """
    Detokenize a binary dataset to a text file.

    Args:
        tokenizer_path: Path to the tokenizer JSON file
        input_dataset_path: Path to the input dataset (without .bin/.idx suffix)
        output_file_path: Path to output text file
    """
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)

    # Input files
    input_data_path = input_dataset_path + ".bin"
    input_index_path = input_dataset_path + ".idx"

    # Check input files exist
    if not os.path.exists(input_data_path) or not os.path.exists(input_index_path):
        print(f"Error: Input dataset files not found: {input_dataset_path}.bin/.idx")
        sys.exit(1)

    # Get number of entries
    input_index_size = os.path.getsize(input_index_path)
    num_entries = input_index_size // 5  # Each index entry is 5 bytes

    print(f"Detokenizing {num_entries:,} entries from {input_dataset_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Statistics
    empty_count = 0

    with open(input_data_path, "rb") as data_file, open(
        input_index_path, "rb"
    ) as index_file, open(output_file_path, "w", encoding="utf-8") as output_file:

        for entry_idx in tqdm(range(num_entries), desc="Detokenizing"):
            try:
                # Read tokens from dataset
                tokens = read_entry(data_file, index_file, entry_idx)

                # Detokenize to text
                text = detokenize_sequence(tokenizer, tokens)
                if not text.strip():
                    empty_count += 1

                output_file.write(text + "\n")

            except Exception as e:
                print(f"Error processing entry {entry_idx}: {e}")
                # Write empty line to maintain correspondence
                output_file.write("\n")
                empty_count += 1

    # Print statistics
    print("✓ Detokenization complete!")
    print(f"  Total entries processed: {num_entries:,}")
    print(f"  Output file: {output_file_path}")
    print(f"  Empty sequences: {empty_count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detokenize binary dataset to readable text file"
    )
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