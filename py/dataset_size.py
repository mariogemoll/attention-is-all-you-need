#!/usr/bin/env python3

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Get the size of a binary dataset")
    parser.add_argument(
        "dataset_file_path_prefix", help="Path to the dataset file (without .bin/.idx suffix)"
    )

    args = parser.parse_args()

    try:
        idx_path = args.dataset_file_path_prefix + ".idx"
        size = os.path.getsize(idx_path)
        print(size // 4)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
