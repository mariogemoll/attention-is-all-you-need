from itertools import islice
from typing import BinaryIO, Dict, Iterator, Tuple

import torch
from tqdm import tqdm

from batching import BucketEntries
from data import open_buckets
from model import Transformer
from params import eos, pad, sos, target_num_tokens_per_batch


class SequenceInfo:
    src_tokens: list[int]
    src_encoding: torch.Tensor
    tgt_tokens: list[int]

    def __init__(self, src_tokens: list[int], src_encoding: torch.Tensor, tgt_tokens: list[int]):
        self.src_tokens = src_tokens
        self.src_encoding = src_encoding
        self.tgt_tokens = tgt_tokens


def add_src_sequences(
    device: torch.device,
    model: Transformer,
    sequences: dict[int, SequenceInfo],
    seq_len: int,
    bucket_iterator: Iterator[Tuple[int, list[int]]],
    num_rows: int,
) -> int:
    batch = list(islice(bucket_iterator, num_rows))
    if not batch:
        return 0
    enc_input = []
    for _, src_tokens in batch:
        # Pad the source tokens to seq_len
        padded_src = src_tokens + [pad] * (seq_len - len(src_tokens))
        enc_input.append(padded_src)
    enc_input_tensor = torch.tensor(enc_input, dtype=torch.long, device=device)
    encoded = model.encode(enc_input_tensor)
    for i in range(len(batch)):
        sequences[batch[i][0]] = SequenceInfo(
            src_tokens=src_tokens, src_encoding=encoded[i], tgt_tokens=[sos]
        )
    return len(batch)


def add_spillover_sequences(
    device: torch.device,
    model: Transformer,
    sequences: dict[int, SequenceInfo],
    seq_len: int,
    spillover_sequences: dict[int, SequenceInfo],
    num_rows: int,
) -> None:
    while len(spillover_sequences) > 0:
        batch = list(islice(spillover_sequences.items(), num_rows))
        if not batch:
            return
        enc_input = []
        for line_number, info in batch:
            # Pad the source tokens to seq_len
            padded_src = info.src_tokens + [pad] * (seq_len - len(info.src_tokens))
            enc_input.append(padded_src)
        enc_input_tensor = torch.tensor(enc_input, dtype=torch.long, device=device)
        encoded = model.encode(enc_input_tensor)
        for i, (line_number, info) in enumerate(batch):
            sequences[line_number] = SequenceInfo(
                src_tokens=info.src_tokens, src_encoding=encoded[i], tgt_tokens=info.tgt_tokens
            )
            spillover_sequences.pop(line_number, None)


def store_output(
    output_index_file: BinaryIO, output_data_file: BinaryIO, entry_idx: int, data: list[int]
) -> None:
    if len(data) == 0:
        pos = 0
    else:
        # Get the current position in the data file
        pos = output_data_file.tell()
        # Convert the data to 16bit little-endian and write to the data file
        output_data_file.write(torch.tensor(data, dtype=torch.int16).numpy().tobytes())

    # The index stores 5 bytes for any entry: a 32bit offset and a 8bit length
    idx_file_pos = entry_idx * 5
    output_index_file.seek(idx_file_pos)
    # Write the starting position as a little-endian 32bit integer
    output_index_file.write(pos.to_bytes(4, byteorder="little"))
    # Write the length of the data as a single byte
    output_index_file.write(len(data).to_bytes(1, byteorder="little"))


def translate_dataset(
    device: torch.device, model: Transformer, input_path_prefix: str, output_path_prefix: str
) -> None:
    model.eval()

    sequences: Dict[int, SequenceInfo] = {}
    spillover_sequences: Dict[int, SequenceInfo] = {}

    with open_buckets(input_path_prefix) as dataset, open(
        output_path_prefix + ".bin", "wb"
    ) as output_data_file, open(
        output_path_prefix + ".idx", "wb"
    ) as output_index_file, torch.no_grad():
        for bucket_idx in range(dataset.num_buckets):
            bucket_entries = BucketEntries(dataset, bucket_id=bucket_idx)
            seq_len = (bucket_idx + 1) * dataset.step_size
            num_rows = target_num_tokens_per_batch // seq_len
            # Make it a multiple of 16
            num_rows = (num_rows // 16) * 16

            num_sequences_completed = 0
            bucket_iterator = iter(bucket_entries)
            iteration_count = 0

            pbar = tqdm(
                total=len(bucket_entries) + len(spillover_sequences), desc=f"Bucket {bucket_idx}"
            )

            # Start with the spillover sequences from the last bucket. Recalculate the encoding
            # tensors so that they have the correct width
            add_spillover_sequences(
                device, model, sequences, seq_len, spillover_sequences, num_rows
            )

            while True:
                iteration_count += 1
                # Try to get num_rows seqences we can process
                batch = dict(islice(sequences.items(), num_rows))
                if not batch or len(batch) < num_rows:
                    num_seqs_added = add_src_sequences(
                        device, model, sequences, seq_len, bucket_iterator, num_rows
                    )
                    if num_seqs_added == 0:
                        if not batch:
                            # There are no sequences to process and there's also no more we can add.
                            # We're done with this bucket.
                            break
                        # If we have some sequences but couldn't add more, process what we have
                    else:
                        # We added sequences, get a fresh batch
                        batch = dict(islice(sequences.items(), num_rows))

                # If we have any zero length sequences in the batch, we just handle those and then
                # run the loop again
                zero_length_seqs = [
                    (line_number, info)
                    for line_number, info in batch.items()
                    if (len(info.src_tokens) == 0)
                ]
                if len(zero_length_seqs) > 0:
                    for line_number, info in zero_length_seqs:
                        # Handle zero length sequences (e.g., by removing them)
                        batch.pop(line_number, None)
                        store_output(output_index_file, output_data_file, line_number - 1, [])
                    print(f"processed {len(zero_length_seqs)} zero length sequences")
                    continue

                enc_input = [info.src_tokens for info in batch.values()]
                memory = [info.src_encoding for info in batch.values()]
                dec_input = [info.tgt_tokens for info in batch.values()]

                enc_input_padded = []
                for src_tokens in enc_input:
                    # Pad the source tokens to seq_len
                    padded_src = src_tokens + [pad] * (seq_len - len(src_tokens))
                    enc_input_padded.append(padded_src)

                dec_input_padded = []
                for tgt_tokens in dec_input:
                    # Pad the target tokens to seq_len
                    padded_tgt = tgt_tokens + [pad] * (seq_len - len(tgt_tokens))
                    dec_input_padded.append(padded_tgt)

                enc_input_tensor = torch.tensor(enc_input_padded, dtype=torch.long, device=device)
                memory_tensor = torch.stack(memory, dim=0)
                dec_input_tensor = torch.tensor(dec_input_padded, dtype=torch.long, device=device)

                result = model.decode(enc_input_tensor, memory_tensor, dec_input_tensor)

                # Go through the result

                for i, (line_number, info) in enumerate(batch.items()):
                    # Look at the newly generated token (at the current length of target tokens)
                    current_tgt_len = len(info.tgt_tokens)

                    new_token = int(result[i][current_tgt_len - 1].argmax().item())

                    if new_token == pad:
                        raise ValueError("Generated padding token")

                    if new_token == eos:
                        # If the sequence ends with eos, it's completely translated. Store the
                        # result and remove the sequence from sequences
                        final_sequence = info.tgt_tokens[1:]  # Remove SOS token
                        store_output(
                            output_index_file, output_data_file, line_number - 1, final_sequence
                        )
                        sequences.pop(line_number, None)
                        num_sequences_completed += 1
                        pbar.update(1)
                    elif current_tgt_len == seq_len:
                        # We've exhausted the sequence length for this bucket (and we haven't
                        # generated eos yet), so we need to move this sequence into the next bucket
                        info.tgt_tokens.append(new_token)
                        spillover_sequences[line_number] = info
                        sequences.pop(line_number, None)
                        num_sequences_completed += 1
                        pbar.update(1)
                    else:
                        # Otherwise, we just add the new token to the list
                        sequences[line_number].tgt_tokens.append(new_token)

            pbar.close()
            print(f"Total sequences completed: {num_sequences_completed}")
            print(f"Spillover sequences: {len(spillover_sequences)}")
        # We're done with all the buckets. If we still have spillover sequences, store the tokens we
        # have generated for those
        for line_number, info in spillover_sequences.items():
            store_output(output_index_file, output_data_file, line_number - 1, info.tgt_tokens[1:])
