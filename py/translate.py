#!/usr/bin/env python3
"""
Translation script for German to English translation using trained transformer model.

Usage:
    python translate.py <model_weights_file> <source_text>
    python translate.py model_epoch_7.pt "Hallo Welt"
    python translate.py checkpoints/model_latest.pt "Das ist ein Test."
    echo "Wie geht es dir?" | python translate.py model_epoch_7.pt
"""

import argparse
import sys

import tokenizers  # type: ignore
import torch

from model import Transformer
from params import eos, max_seq_len, pad, sos
from util import get_device


def load_tokenizer(tokenizer_path: str = "../3_tokenizer/tokenizer.json") -> tokenizers.Tokenizer:
    """Load the BPE tokenizer."""
    try:
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer from {tokenizer_path}: {e}")
        sys.exit(1)


def tokenize_source(tokenizer: tokenizers.Tokenizer, text: str) -> torch.Tensor:
    """Tokenize source text and prepare for model input."""
    # Tokenize the input text
    tokens = tokenizer.encode(text).ids

    # Check length limit
    if len(tokens) > max_seq_len - 2:  # Leave room for SOS/EOS if needed
        print(
            f"Warning: Input text too long ({len(tokens)} tokens), truncating to {max_seq_len - 2}"
        )
        tokens = tokens[: max_seq_len - 2]

    # Convert to tensor and add batch dimension
    src_tensor = torch.tensor([tokens], dtype=torch.long)
    return src_tensor


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_length: int = max_seq_len,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate translation using greedy decoding."""
    model.eval()
    src = src.to(device)

    # Encode source sequence
    with torch.no_grad():
        memory = model.encode(src)

        # Start with SOS token
        tgt = torch.tensor([[sos]], device=device, dtype=torch.long)

        for _ in range(max_length - 1):
            # Get model output for current target sequence
            out = model.decode(src, memory, tgt)

            # Get the last token's probabilities and choose the most likely
            next_token = out[:, -1, :].argmax(dim=-1).unsqueeze(1)

            # Stop if we generate EOS token
            if next_token.item() == eos:
                break

            # Append the predicted token to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)

    return tgt


def beam_search_decode(
    model: Transformer,
    src: torch.Tensor,
    beam_size: int = 4,
    max_length: int = max_seq_len,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate translation using beam search decoding."""
    model.eval()
    src = src.to(device)

    with torch.no_grad():
        memory = model.encode(src)

        # Initialize beam with SOS token
        # beam: list of (sequence, log_prob)
        beam = [(torch.tensor([sos], device=device), 0.0)]

        for step in range(max_length - 1):
            candidates = []

            for seq, score in beam:
                if seq[-1].item() == eos:
                    # If sequence already ended, just add it to candidates
                    candidates.append((seq, score))
                    continue

                # Prepare input for model
                tgt_input = seq.unsqueeze(0)  # Add batch dimension

                # Get model predictions
                out = model.decode(src, memory, tgt_input)
                log_probs = torch.log_softmax(out[:, -1, :], dim=-1)

                # Get top beam_size predictions
                top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                for i in range(beam_size):
                    token = top_indices[0, i].item()
                    token_log_prob = top_log_probs[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([token], device=device)])
                    new_score = score + token_log_prob
                    candidates.append((new_seq, new_score))

            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]

            # Check if all sequences have ended
            if all(seq[-1].item() == eos for seq, _ in beam):
                break

        # Return the best sequence
        best_seq, _ = beam[0]
        return best_seq.unsqueeze(0)  # Add batch dimension


def detokenize_output(tokenizer: tokenizers.Tokenizer, tokens: torch.Tensor) -> str:
    """Convert model output tokens back to text."""
    # Remove batch dimension and convert to list
    tokens_list = tokens[0].tolist()

    # Remove SOS token if present
    if tokens_list and tokens_list[0] == sos:
        tokens_list = tokens_list[1:]

    # Remove EOS token if present
    if tokens_list and tokens_list[-1] == eos:
        tokens_list = tokens_list[:-1]

    # Remove PAD tokens
    tokens_list = [token for token in tokens_list if token != pad]

    # Decode to text
    try:
        text: str = str(tokenizer.decode(tokens_list))
        return text
    except Exception as e:
        print(f"Warning: Error detokenizing: {e}")
        # Fallback: try to decode each token individually
        words = []
        for token in tokens_list:
            try:
                word: str = str(tokenizer.decode([token]))
                words.append(word)
            except Exception:
                words.append(f"<UNK_{token}>")
        return "".join(words)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate German text to English")
    parser.add_argument(
        "model_weights_file",
        help="Path to model weights file (e.g., model_epoch_7.pt, checkpoints/model_latest.pt)",
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
