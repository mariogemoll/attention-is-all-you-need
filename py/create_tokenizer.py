import sys

from tokenization import create_tokenizer

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <output_file> [<file1> ...]")
        sys.exit(1)
    output_file_path = sys.argv[1]
    files = sys.argv[2:]

    create_tokenizer(output_file_path, files)
