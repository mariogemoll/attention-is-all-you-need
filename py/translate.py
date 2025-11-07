#!/usr/bin/env python3
"""
Translation script for German to English translation using trained transformer model.

Usage:
    python translate.py <model_weights_file> <source_text>
    python translate.py model_0007.pt "Hallo Welt"
    echo "Wie geht es dir?" | python translate.py model_0007.pt
"""

import argparse
import sys

import torch

from model import Transformer
from params import max_seq_len
from translation import (
    beam_search_decode,
    detokenize_output,
    greedy_decode,
    load_tokenizer,
    tokenize_source,
)
from util import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate German text to English")
    parser.add_argument(
        "model_weights_file",
        help="Path to model weights file (e.g., model_0007.pt)",
    )
    parser.add_argument(
        "source_text",
        nargs="?",
        help="German text to translate (if not provided, reads from stdin)",
    )
    parser.add_argument(
        "--tokenizer",
        default="../3_tokenizer/tokenizer.json",
        help="Path to tokenizer file (default: ../3_tokenizer/tokenizer.json)",
    )
    parser.add_argument(
        "--beam-search", action="store_true", help="Use beam search instead of greedy decoding"
    )
    parser.add_argument(
        "--beam-size", type=int, default=4, help="Beam size for beam search (default: 4)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=max_seq_len,
        help=f"Maximum translation length (default: {max_seq_len})",
    )

    args = parser.parse_args()

    # Get input text
    if args.source_text:
        source_text = args.source_text
    else:
        # Read from stdin
        print("Enter German text to translate:")
        source_text = sys.stdin.read().strip()

    if not source_text:
        print("Error: No input text provided")
        sys.exit(1)

    print(f"Source: {source_text}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    print(f"Loading model from {args.model_weights_file}...")
    device = get_device()
    model = Transformer().to(device)

    try:
        model_state_dict = torch.load(args.model_weights_file, map_location=device)
        model.load_state_dict(model_state_dict)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Tokenize input
    print("Tokenizing input...")
    src_tokens = tokenize_source(tokenizer, source_text)
    print(f"Source tokens: {src_tokens[0].tolist()}")

    # Generate translation
    print("Generating translation...")
    if args.beam_search:
        print(f"Using beam search (beam_size={args.beam_size})")
        output_tokens = beam_search_decode(
            model, src_tokens, beam_size=args.beam_size, max_length=args.max_length, device=device
        )
    else:
        print("Using greedy decoding")
        output_tokens = greedy_decode(model, src_tokens, max_length=args.max_length, device=device)

    print(f"Output tokens: {output_tokens[0].tolist()}")

    # Detokenize output
    translation = detokenize_output(tokenizer, output_tokens)

    print("\n" + "=" * 50)
    print(f"Translation: {translation}")
    print("=" * 50)


if __name__ == "__main__":
    main()
