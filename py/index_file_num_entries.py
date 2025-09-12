import os
import sys


def print_index_file_info(filename: str) -> None:
    try:
        size = os.path.getsize(filename)
        if size % 4 != 0:
            print(f"Warning: File size {size} is not a multiple of 4 bytes. File may be corrupted.")
        num_entries = size // 4
        print(num_entries)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python indexed_file_info.py <index_file>")
        sys.exit(1)
    print_index_file_info(sys.argv[1])
