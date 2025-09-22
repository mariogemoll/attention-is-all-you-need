import sys

import torch

from inference import translate_dataset
from model import Transformer
from util import get_device


def main() -> None:
    if len(sys.argv) not in (4, 5):
        print(f"Usage: {sys.argv[0]} <model_path> <input_path> <output_path> [beam_size]")
        sys.exit(1)

    beam_size = 4 if len(sys.argv) == 4 else int(sys.argv[4])

    device = get_device()
    print(f"Using device: {device}")
    model = Transformer()
    model.load_state_dict(torch.load(sys.argv[1], map_location=device))
    model.to(device)
    model.eval()

    # Open the dataset
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    translate_dataset(device, model, input_path, output_path, beam_size=beam_size)


if __name__ == "__main__":
    main()
