import sys

import torch

from inference import translate_dataset
from model import Transformer
from util import get_device


def main() -> None:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <model_path> <input_path> <output_path>")
        sys.exit(1)

    device = get_device()
    model = Transformer()
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.to(device)
    model.eval()

    # Open the dataset
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    translate_dataset(device, model, input_path, output_path)


if __name__ == "__main__":
    main()
