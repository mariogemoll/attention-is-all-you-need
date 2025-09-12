import os
import sys

from buckets import create_bucket_index
from params import max_parallelism, max_seq_len


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} dataset_path_prefix")
        sys.exit(1)

    dataset_path_prefix = sys.argv[1]

    try:
        num_procs = min(os.cpu_count() or 1, max_parallelism)
        create_bucket_index(
            dataset_path_prefix, step_size=16, max_length=max_seq_len, num_processes=num_procs
        )
    except Exception as e:
        print(f"❌ Error during index creation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
