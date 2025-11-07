# type: ignore

import sys

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: python upload_tokenizer.py <path-to-tokenizer> <huggingface-repo-name> "
            "<commit-message>"
        )
        sys.exit(1)
    tokenizer = Tokenizer.from_file(sys.argv[1])
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    hf_tokenizer.push_to_hub(sys.argv[2], commit_message=sys.argv[3])


if __name__ == "__main__":
    main()
