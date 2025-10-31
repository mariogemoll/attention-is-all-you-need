#!/usr/bin/env python3
"""
Export a specific metric from TensorBoard events file to binary format.
"""
import struct
import sys
from pathlib import Path
from typing import List

from read_tensorboard import read_tensorboard_events


def export_metric_to_binary(events_file: str, metric_label: str, output_file: str) -> None:
    """
    Export a specific metric from TensorBoard events to binary file.

    Args:
        events_file: Path to the TensorBoard events file
        metric_label: The metric label/tag to export
        output_file: Path to the output binary file

    Raises:
        ValueError: If the metric label is not found in the events file
    """
    # Read the TensorBoard data
    data = read_tensorboard_events(events_file)

    # Check if the metric exists
    if metric_label not in data:
        available_metrics = sorted(data.keys())
        raise ValueError(
            f"Metric '{metric_label}' not found in events file.\n"
            f"Available metrics: {', '.join(available_metrics)}"
        )

    # Extract the values (third element in each tuple)
    values: List[float] = [v[2] for v in data[metric_label]]

    # Write as little-endian 32-bit floats
    with open(output_file, "wb") as f:
        for value in values:
            # '<f' means little-endian 32-bit float
            f.write(struct.pack("<f", value))

    print(f"Exported {len(values)} values from '{metric_label}' to {output_file}")
    print(f"File size: {len(values) * 4} bytes")


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python export_metric_binary.py <events_file> <metric_label> <output_file>")
        print("\nExample:")
        print(
            "  python export_metric_binary.py "
            "../runs/ddp_20251031_123913/events.out.tfevents.* "
            "loss/batch/train output.bin"
        )
        print("\nTo see available metrics, run:")
        print("  python read_tensorboard.py <events_file>")
        sys.exit(1)

    events_file = sys.argv[1]
    metric_label = sys.argv[2]
    output_file = sys.argv[3]

    if not Path(events_file).exists():
        print(f"Error: Events file not found: {events_file}")
        sys.exit(1)

    try:
        export_metric_to_binary(events_file, metric_label, output_file)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
