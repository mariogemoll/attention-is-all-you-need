import os
import sys

from dataset import create_subset

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            f"Usage: python {sys.argv[0]} <input_file_path_prefix> <output_file_path_prefix> "
            f"<num_samples>"
        )
        print()
        print("Example: python create_random_subset.py ../4_tokens/train sample_1000 1000")
        print("This will create a random subset of 1000 samples from the train dataset.")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    try:
        num_samples = int(sys.argv[3])
    except ValueError:
        print(f"Error: num_samples must be an integer, got '{sys.argv[3]}'")
        sys.exit(1)

    if num_samples <= 0:
        print(f"Error: num_samples must be positive, got {num_samples}")
        sys.exit(1)

    # Check if input files exist
    required_files = [
        f"{input_file_path}.src.idx",
        f"{input_file_path}.src.bin",
        f"{input_file_path}.tgt.idx",
        f"{input_file_path}.tgt.bin",
        f"{input_file_path}.meta",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: Missing input files:")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)

    print(f"Sampling {num_samples} random entries from {input_file_path}")
    print(f"Output will be saved to {output_file_path}")

    try:
        create_subset(input_file_path, output_file_path, num_samples)
        print("Random subset sampling completed successfully!")
    except Exception as e:
        print(f"Error during sampling: {e}")
        sys.exit(1)
