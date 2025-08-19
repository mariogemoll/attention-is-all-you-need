import sys

from bleu import get_bleu_score


def run() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <reference_file> <translation_file>")
        sys.exit(1)

    reference_file_path = sys.argv[1]
    translation_file_path = sys.argv[2]

    bleu_score = get_bleu_score(reference_file_path, translation_file_path)
    print(f"BLEU score: {bleu_score}")


if __name__ == "__main__":
    run()
