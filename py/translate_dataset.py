import argparse
import os
import sys

import torch

from inference import translate_dataset
from model import Transformer
from util import get_device


def main() -> None:
    # Check if using old positional arguments
    if len(sys.argv) >= 4 and not any(arg.startswith("-") for arg in sys.argv[1:4]):
        # Old style: python translate_dataset.py model input output [beam_size] [--show-progress]
        model_path = sys.argv[1]
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        beam_size = 4 if len(sys.argv) <= 4 else int(sys.argv[4])
        show_progress = "--show-progress" in sys.argv[4:]
    else:
        # New argparse style
        parser = argparse.ArgumentParser(description="Translate dataset using a trained model")
        parser.add_argument("model_path", help="Path to the model file")
        parser.add_argument("input_path", help="Path to input dataset")
        parser.add_argument("output_path", help="Path to output dataset")
        parser.add_argument(
            "beam_size", nargs="?", type=int, default=4, help="Beam size for translation"
        )
        parser.add_argument(
            "--show-progress", action="store_true", help="Show progress bars during translation"
        )

        args = parser.parse_args()
        model_path = args.model_path
        input_path = args.input_path
        output_path = args.output_path
        beam_size = args.beam_size
        show_progress = args.show_progress

    device = get_device()
    model_name = os.path.basename(model_path)

    # Always print which model we're working on
    print(f"Translating with model: {model_name}")
    if not show_progress:
        print(f"Using device: {device}")

    model = Transformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Translate the dataset
    translate_dataset(
        device=device,
        model=model,
        input_path_prefix=input_path,
        output_path_prefix=output_path,
        beam_size=beam_size,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    main()
