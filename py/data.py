from contextlib import contextmanager
from typing import BinaryIO, Generator

import torch

from params import eos, pad, sos
from serialization import get_entries, read_bucket_index_header


class BucketedDataset:
    bucket_index_file: BinaryIO
    step_size: int
    num_buckets: int
    index_file: BinaryIO
    data_file: BinaryIO
    data_file_size: int

    def __init__(
        self,
        bucket_index_file: BinaryIO,
        step_size: int,
        num_buckets: int,
        index_file: BinaryIO,
        data_file: BinaryIO,
        data_file_size: int,
    ):
        self.bucket_index_file = bucket_index_file
        self.step_size = step_size
        self.num_buckets = num_buckets
        self.index_file = index_file
        self.data_file = data_file
        self.data_file_size = data_file_size


@contextmanager
def open_buckets(base_path: str) -> Generator[BucketedDataset, None, None]:
    bucket_index_file = open(f"{base_path}.bidx", "rb")
    idx_file = open(f"{base_path}.idx", "rb")
    data_file = open(f"{base_path}.bin", "rb")
    try:
        data_file.seek(0, 2)
        data_file_size = data_file.tell()
        step_size, num_buckets, _ = read_bucket_index_header(bucket_index_file)
        yield BucketedDataset(
            bucket_index_file=bucket_index_file,
            step_size=step_size,
            num_buckets=num_buckets,
            index_file=idx_file,
            data_file=data_file,
            data_file_size=data_file_size,
        )
    finally:
        bucket_index_file.close()
        idx_file.close()
        data_file.close()


def get_tensors(
    idx_file: BinaryIO, data_file: BinaryIO, data_file_length: int, seq_len: int, ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    enc_input_tensor = torch.tensor(enc_input, dtype=torch.int64)
    dec_input_tensor = torch.tensor(dec_input, dtype=torch.int64)
    dec_target_tensor = torch.tensor(dec_target, dtype=torch.int64)

    return enc_input_tensor, dec_input_tensor, dec_target_tensor
