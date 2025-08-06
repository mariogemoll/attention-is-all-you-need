import sys

from serialization import combine_datasets


def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} output_dataset input_dataset1 [input_dataset2 ...]")
        print()
        print("Examples:")
        print(f"  python {sys.argv[0]} combined_dataset europarl newscommentary commoncrawl")
        print(f"  python {sys.argv[0]} /path/to/output /path/to/dataset1 /path/to/dataset2")
        print()
        print("Note: Do not include .idx suffix - it will be handled automatically")
        sys.exit(1)

    output_path = sys.argv[1]
    input_paths = sys.argv[2:]

    try:
        combine_datasets(output_path, input_paths)
    except Exception as e:
        print(f"❌ Error during concatenation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
