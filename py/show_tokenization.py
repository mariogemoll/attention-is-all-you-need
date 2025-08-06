import sys

import tokenizers  # type: ignore

from tokenization import test_tokenizer

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f'Usage: python {sys.argv[0]} <tokenizer_json> "<text>"')
        sys.exit(1)

    tokenizer = tokenizers.Tokenizer.from_file(sys.argv[1])

    test_tokenizer(tokenizer, sys.argv[2])
