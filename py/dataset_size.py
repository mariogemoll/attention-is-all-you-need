#!/usr/bin/env python3

import argparse

from serialization import get_number_of_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Get the size of a binary dataset")
    parser.add_argument(
        "dataset_file_path_prefix", help="Path to the dataset file (without .bin/.idx suffix)"
    )

    args = parser.parse_args()

    try:
        num_entries = get_number_of_entries(args.dataset_file_path_prefix)
        print(num_entries)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
