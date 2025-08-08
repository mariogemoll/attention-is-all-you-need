import sys

from params import max_seq_len
from serialization import create_bucket_index


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} dataset_path_prefix")
        sys.exit(1)

    dataset_path_prefix = sys.argv[1]

    try:
        create_bucket_index(dataset_path_prefix, step_size=16, max_length=max_seq_len)
    except Exception as e:
        print(f"❌ Error during index creation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
