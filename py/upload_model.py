import sys

import torch

from model import Transformer


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python upload.py <path-to-weights> <huggingface-repo-name> <commit-message>")
        sys.exit(1)
    model = Transformer()
    model_state_dict = torch.load(sys.argv[1])
    model.load_state_dict(model_state_dict)
    model.save_pretrained(sys.argv[2])
    model.push_to_hub(sys.argv[2], commit_message=sys.argv[3])


if __name__ == "__main__":
    main()
