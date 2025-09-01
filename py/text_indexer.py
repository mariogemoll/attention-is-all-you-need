#!/usr/bin/env python3
"""
Text file indexer that generates a binary index file storing the end positions of each line.
Each position is stored as a 4-byte little-endian integer.
"""

import os
import struct
import sys


def generate_index(input_file_path: str, index_file_path: str | None = None) -> str:
    """
    Generate a binary index file that stores the end position of each line as 4-byte little-endian
    integers.

    Args:
        input_file_path (str): Path to the input text file
        index_file_path (str): Path to the output binary index file (optional)

    Returns:
        str: Path to the generated index file
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    if index_file_path is None:
        index_file_path = input_file_path + ".idx"

    line_end_positions = []

    with open(input_file_path, "rb") as f:
        position = 0
        while True:
            line = f.readline()
            if not line:
                break
            position += len(line)
            line_end_positions.append(position - 1)

    with open(index_file_path, "wb") as idx_file:
        for pos in line_end_positions:
            idx_file.write(struct.pack("<I", pos))

    return index_file_path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python text_indexer.py <input_file> [index_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    index_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result_index_file = generate_index(input_file, index_file)
        print(f"Binary index file generated: {result_index_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
