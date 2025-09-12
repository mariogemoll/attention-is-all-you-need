"""Helper functions for testing dataset-related functionality."""

import struct

from dataset import Dataset, append_to_dataset


def make_toy_tokens(seed: int, n: int) -> list[int]:
    """Create test tokens with a predictable pattern."""
    return [(seed + i) % 65536 for i in range(n)]


def make_toy_dataset(prefix: str, entries: list[tuple[list[int], list[int]]]) -> None:
    """Create a test dataset with the given entries.

    Args:
        prefix: Base path for dataset files (without extensions)
        entries: List of (src_tokens, tgt_tokens) tuples
    """
    with (
        open(prefix + ".src.idx", "wb+") as src_idx,
        open(prefix + ".src.bin", "wb+") as src_bin,
        open(prefix + ".tgt.idx", "wb+") as tgt_idx,
        open(prefix + ".tgt.bin", "wb+") as tgt_bin,
        open(prefix + ".meta", "wb+") as meta,
    ):
        dataset = Dataset(src_idx, src_bin, tgt_idx, tgt_bin, meta, 0, 0, 0)
        for i, (src_tokens, tgt_tokens) in enumerate(entries):
            append_to_dataset(
                dataset,
                corpus_id=0,
                original_line_number=i,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
            )


def make_sequential_single_stream_dataset(prefix: str, entries: list[list[int]]) -> None:
    """Create a sequential single-stream dataset (.idx 4B ends, .bin 2B tokens).

    Args:
        prefix: Base path without extension
        entries: List of token ID lists (each entry becomes a record)
    """
    bin_path = prefix + ".bin"
    idx_path = prefix + ".idx"
    cumulative = 0
    with open(bin_path, "wb") as b, open(idx_path, "wb") as idx:
        for tokens in entries:
            if tokens:
                b.write(struct.pack(f"<{len(tokens)}H", *tokens))
            cumulative += len(tokens) * 2
            idx.write(struct.pack("<I", cumulative))
