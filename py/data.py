from contextlib import contextmanager
from typing import BinaryIO, Generator, Tuple

import numpy as np

from params import eos, pad, sos
from serialization import get_entries


@contextmanager
def open_buckets(base_path: str) -> Generator[Tuple[BinaryIO, BinaryIO, BinaryIO, int], None, None]:
    """
    Context manager to open dataset files (bucket index, index, and data files).

    Args:
        base_path: Path to the dataset files without extension (e.g., "../4_tokens/train")

    Yields:
        Tuple of (bucket_index_file, idx_file, data_file, data_file_size)
    """
    bucket_index_file = open(f"{base_path}.bidx", "rb")
    idx_file = open(f"{base_path}.idx", "rb")
    data_file = open(f"{base_path}.bin", "rb")

    try:
        # Get data file size
        data_file.seek(0, 2)
        data_file_size = data_file.tell()

        yield bucket_index_file, idx_file, data_file, data_file_size
    finally:
        bucket_index_file.close()
        idx_file.close()
        data_file.close()


def get_tensors(
    idx_file: BinaryIO, data_file: BinaryIO, data_file_length: int, seq_len: int, ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entries = get_entries(idx_file, data_file, data_file_length, ids)

    enc_input = []
    dec_input = []
    dec_target = []

    # Prepare tensors for encoder input, decoder input, and decoder target padded to seq_len
    for _, _, src_tokens, tgt_tokens in entries:
        enc_input.append(src_tokens + [pad] * (seq_len - len(src_tokens)))
        # Add start of sequence token to decoder input and end of sequence token to decoder target
        dec_input.append([sos] + tgt_tokens + [pad] * (seq_len - 1 - len(tgt_tokens)))
        dec_target.append(tgt_tokens + [eos] + [pad] * (seq_len - 1 - len(tgt_tokens)))

    enc_input_tensor = np.array(enc_input, dtype=np.int64)
    dec_input_tensor = np.array(dec_input, dtype=np.int64)
    dec_target_tensor = np.array(dec_target, dtype=np.int64)

    return enc_input_tensor, dec_input_tensor, dec_target_tensor
