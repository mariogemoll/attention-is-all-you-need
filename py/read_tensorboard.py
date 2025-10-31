#!/usr/bin/env python3
"""
Read TensorBoard events file and extract metrics data.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from tensorboard.backend.event_processing import event_accumulator  # type: ignore[import-untyped]


def read_tensorboard_events(events_file: str) -> Dict[str, List[Tuple[int, float, float]]]:
    """
    Read TensorBoard events file and extract all scalar metrics.

    Args:
        events_file: Path to the TensorBoard events file

    Returns:
        Dictionary mapping metric names to lists of (step, wall_time, value) tuples
    """
    # Create an event accumulator
    ea = event_accumulator.EventAccumulator(
        events_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 means load all
        },
    )

    # Load the events
    ea.Reload()

    # Get all available scalar tags
    tags = ea.Tags()["scalars"]

    # Extract data for each tag
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        # Store as list of (step, wall_time, value)
        data[tag] = [(e.step, e.wall_time, e.value) for e in events]

    return data


def print_summary(data: Dict[str, List[Tuple[int, float, float]]]) -> None:
    """Print a summary of the metrics found in the events file."""
    print(f"Found {len(data)} metrics:")
    print()

    for tag, values in sorted(data.items()):
        if values:
            steps = [v[0] for v in values]
            metric_values = [v[2] for v in values]
            print(f"  {tag}:")
            print(f"    - Number of data points: {len(values)}")
            print(f"    - Step range: {min(steps)} to {max(steps)}")
            print(f"    - Value range: {min(metric_values):.6f} to {max(metric_values):.6f}")
            print(f"    - Latest value: {metric_values[-1]:.6f}")
            print()


def main() -> Dict[str, List[Tuple[int, float, float]]]:
    if len(sys.argv) < 2:
        print("Usage: python read_tensorboard.py <events_file>")
        print("\nExample:")
        print("  python read_tensorboard.py ../runs/ddp_20251031_123913/events.out.tfevents.*")
        sys.exit(1)

    events_file = sys.argv[1]

    if not Path(events_file).exists():
        print(f"Error: File not found: {events_file}")
        sys.exit(1)

    print(f"Reading TensorBoard events from: {events_file}")
    print()

    # Read the events
    data = read_tensorboard_events(events_file)

    # Print summary
    print_summary(data)

    # Return the data for potential import/reuse
    return data


if __name__ == "__main__":
    data = main()
