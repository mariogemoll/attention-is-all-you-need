import torch

from dataset import Dataset
from params import eos, pad, sos


def get_tensors(
    seq_len: int, dataset: "Dataset", entry_indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from dataset import get_entry

    entries = [get_entry(dataset, idx) for idx in entry_indices]
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
