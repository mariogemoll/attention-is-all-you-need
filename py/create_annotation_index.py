#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path


def create_annotation_index(annotation_file: Path, output_file: Path) -> None:
    """
    Create an annotation index file with 4-byte little-endian integers.

    Each sentence pair gets mapped to its corresponding annotation line index.

    Args:
        annotation_file: Path to the .annotation file
        output_file: Path to write the binary index file
    """
    print(f"Reading annotation file: {annotation_file}")

    # Parse annotation file to build mapping from sentence indices to annotation line numbers
    sentence_to_annotation = {}

    with open(annotation_file, "r", encoding="utf-8") as f:
        for annotation_line_idx, line in enumerate(f, 0):  # 0-based line numbers
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue

            src_url, tgt_url, start_idx_str, count_str = parts

            try:
                start_idx = int(start_idx_str)
                sentence_count = int(count_str)
            except ValueError:
                continue

            # Map each sentence pair index to this annotation line
            for sentence_idx in range(start_idx, start_idx + sentence_count):
                sentence_to_annotation[sentence_idx] = annotation_line_idx

    print(f"Found {len(sentence_to_annotation)} sentence pairs mapped to annotations")

    # Find the maximum sentence index to determine output file size
    max_sentence_idx = max(sentence_to_annotation.keys()) if sentence_to_annotation else 0
    total_sentences = max_sentence_idx + 1

    print(f"Creating index file for {total_sentences} sentence pairs")

    # Write binary index file
    with open(output_file, "wb") as f:
        for sentence_idx in range(total_sentences):
            annotation_line = sentence_to_annotation.get(sentence_idx, 0)
            # Pack as 4-byte little-endian unsigned integer
            f.write(struct.pack("<I", annotation_line))

    print(f"Index file written to: {output_file}")
    print(f"File size: {output_file.stat().st_size} bytes ({total_sentences * 4} expected)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create annotation index file for sentence pairs")
    parser.add_argument("annotation_file", type=Path, help="Input annotation file")
    parser.add_argument("output_file", type=Path, help="Output binary index file")

    args = parser.parse_args()

    if not args.annotation_file.exists():
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return 1

    create_annotation_index(args.annotation_file, args.output_file)
    return 0


if __name__ == "__main__":
    exit(main())
