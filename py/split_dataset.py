import sys

from dataset import split_dataset

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            f"Usage: python {sys.argv[0]} <input_file_path_prefix> <output_file_path_prefix_a> "
            f"<output_file_path_prefix_b> <num_samples_in_a>"
        )
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_a_path = sys.argv[2]
    output_file_b_path = sys.argv[3]
    num_samples_in_a = int(sys.argv[4])

    split_dataset(input_file_path, output_file_a_path, output_file_b_path, num_samples_in_a)
