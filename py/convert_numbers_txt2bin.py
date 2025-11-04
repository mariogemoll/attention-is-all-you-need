"""
Export numeric values from a text file to binary format.
"""

import struct
import sys
from pathlib import Path
from typing import List


def export_text_to_binary(input_file: str, output_file: str, skip_header: bool = True) -> None:
    """
    Export numeric values from a text file to binary file.

    Args:
        input_file: Path to the input text file (one value per line)
        output_file: Path to the output binary file
        skip_header: Whether to skip the first line (header)

    Raises:
        ValueError: If a line cannot be converted to float
        FileNotFoundError: If the input file doesn't exist
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    values: List[float] = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header if requested
    start_line = 1 if skip_header and lines else 0

    for i, line in enumerate(lines[start_line:], start=start_line + 1):
        line = line.strip()
        if not line:  # Skip empty lines
            continue

        try:
            value = float(line)
            values.append(value)
        except ValueError as e:
            raise ValueError(f"Cannot convert line {i} to float: '{line}'. Error: {e}")

    if not values:
        raise ValueError("No valid numeric values found in the input file")

    # Write as little-endian 32-bit floats
    with open(output_file, "wb") as f:
        for value in values:
            # '<f' means little-endian 32-bit float
            f.write(struct.pack("<f", value))

    print(f"Exported {len(values)} values from '{input_file}' to '{output_file}'")
    print(f"File size: {len(values) * 4} bytes")
    if values:
        print(f"Value range: {min(values):.6f} to {max(values):.6f}")


def main() -> None:
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python export_text_to_binary.py <input_file> <output_file> [--no-header]")
        print("\nExamples:")
        print("  python export_text_to_binary.py all2013.txt all2013.bin")
        print("  python export_text_to_binary.py data.txt output.bin --no-header")
        print("\nBy default, the first line is skipped (assumed to be a header).")
        print("Use --no-header to process all lines including the first one.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    skip_header = True

    # Check for --no-header flag
    if len(sys.argv) == 4:
        if sys.argv[3] == "--no-header":
            skip_header = False
        else:
            print(f"Error: Unknown flag '{sys.argv[3]}'. Use --no-header to process all lines.")
            sys.exit(1)

    try:
        export_text_to_binary(input_file, output_file, skip_header)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
