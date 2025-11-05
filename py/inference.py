from dataclasses import dataclass
from itertools import islice
from typing import BinaryIO, Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from batching import BucketEntries
from buckets import open_buckets
from data import to_uint16_le_bytes
from indexed_sequential import append_indexed_entry
from model import Transformer
from params import eos, inference_target_num_tokens_per_batch, pad, sos


@dataclass
class BeamState:
    tokens: List[int]
    log_prob: float
    ended: bool = False


@dataclass
class SequenceInfo:
    src_tokens: List[int]
    src_encoding: torch.Tensor
    beams: List[BeamState]


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
        line_number, src_tokens = batch[i]
        sequences[line_number] = SequenceInfo(
            src_tokens=src_tokens,
            src_encoding=encoded[i],
            beams=[BeamState(tokens=[sos], log_prob=0.0, ended=False)],
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
            info.src_encoding = encoded[i]
            sequences[line_number] = info
            spillover_sequences.pop(line_number, None)


def store_output(output_index_file: BinaryIO, output_data_file: BinaryIO, data: list[int]) -> None:
    entry_bytes = to_uint16_le_bytes(data)
    append_indexed_entry(output_data_file, output_index_file, entry_bytes)


def translate_dataset(
    device: torch.device,
    model: Transformer,
    input_path_prefix: str,
    output_path_prefix: str,
    beam_size: int = 4,
    show_progress: bool = True,
) -> None:
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")

    model.eval()

    sequences: Dict[int, SequenceInfo] = {}
    spillover_sequences: Dict[int, SequenceInfo] = {}
    completed_sequences: Dict[int, list[int]] = {}

    with open_buckets(input_path_prefix) as dataset, torch.no_grad():
        for bucket_idx in range(dataset.num_buckets):
            bucket_entries = BucketEntries(dataset, bucket_id=bucket_idx)
            seq_len = (bucket_idx + 1) * dataset.step_size
            num_rows = inference_target_num_tokens_per_batch // seq_len
            # Make it a multiple of 16
            num_rows = (num_rows // 16) * 16

            num_sequences_completed = 0
            bucket_iterator = iter(bucket_entries)

            if show_progress:
                pbar = tqdm(
                    total=len(bucket_entries) + len(spillover_sequences),
                    desc=f"Bucket {bucket_idx}",
                )
            else:
                pbar = None

            # Start with the spillover sequences from the last bucket. Recalculate the encoding
            # tensors so that they have the correct width
            add_spillover_sequences(
                device, model, sequences, seq_len, spillover_sequences, num_rows
            )

            while True:
                # Try to get num_rows sequences we can process
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

                zero_length_sequences = [
                    line_number for line_number, info in batch.items() if len(info.src_tokens) == 0
                ]
                if zero_length_sequences:
                    for line_number in zero_length_sequences:
                        completed_sequences[line_number] = []
                        sequences.pop(line_number, None)
                    num_sequences_completed += len(zero_length_sequences)
                    if pbar is not None:
                        pbar.update(len(zero_length_sequences))
                    print(f"processed {len(zero_length_sequences)} zero length sequences")
                    continue

                beam_entries: List[Tuple[int, int, BeamState]] = []
                sequence_candidates: Dict[int, List[BeamState]] = {
                    line_number: [] for line_number in batch.keys()
                }

                # Pre-allocate tensors filled with padding
                enc_input_padded: List[List[int]] = [[pad] * seq_len for _ in range(num_rows)]
                dec_input_padded: List[List[int]] = [[pad] * seq_len for _ in range(num_rows)]
                memory_tensors: List[torch.Tensor] = []

                # Collect active beam entries up to num_rows
                entry_idx = 0
                for line_number, info in batch.items():
                    if entry_idx >= num_rows:
                        break
                    padded_src = info.src_tokens + [pad] * (seq_len - len(info.src_tokens))
                    for beam_idx, beam_state in enumerate(info.beams):
                        if entry_idx >= num_rows:
                            break
                        if beam_state.ended:
                            sequence_candidates[line_number].append(beam_state)
                            continue
                        if len(beam_state.tokens) > seq_len:
                            # Tokens are already longer than this bucket allows; handle spillover
                            sequence_candidates[line_number].append(beam_state)
                            continue
                        beam_entries.append((line_number, beam_idx, beam_state))
                        enc_input_padded[entry_idx] = padded_src
                        memory_tensors.append(info.src_encoding)
                        padded_tgt = beam_state.tokens + [pad] * (seq_len - len(beam_state.tokens))
                        dec_input_padded[entry_idx] = padded_tgt
                        entry_idx += 1

                # Fill remaining memory tensors with dummy values if needed
                if beam_entries:
                    num_active = len(beam_entries)
                    # Round up to nearest multiple of 16
                    effective_batch_size = ((num_active + 15) // 16) * 16

                    # Resize tensors to effective_batch_size
                    enc_input_padded = enc_input_padded[:effective_batch_size]
                    dec_input_padded = dec_input_padded[:effective_batch_size]

                    dummy_memory = memory_tensors[0]
                    while len(memory_tensors) < effective_batch_size:
                        memory_tensors.append(dummy_memory)

                if beam_entries:
                    enc_input_tensor = torch.tensor(
                        enc_input_padded, dtype=torch.long, device=device
                    )
                    memory_tensor = torch.stack(memory_tensors, dim=0)
                    dec_input_tensor = torch.tensor(
                        dec_input_padded, dtype=torch.long, device=device
                    )
                    decode_output = model.decode(enc_input_tensor, memory_tensor, dec_input_tensor)
                else:
                    decode_output = None

                if decode_output is not None:
                    for idx, (line_number, _beam_idx, beam_state) in enumerate(beam_entries):
                        current_length = len(beam_state.tokens)
                        logits = decode_output[idx][current_length - 1]
                        log_probs = F.log_softmax(logits, dim=0)
                        top_log_probs, top_indices = torch.topk(log_probs, beam_size)

                        # Convert to CPU numpy arrays once for all beam_size values
                        top_indices_cpu = top_indices.cpu().numpy()
                        top_log_probs_cpu = top_log_probs.cpu().numpy()

                        expanded = False
                        for j in range(beam_size):
                            token = int(top_indices_cpu[j])
                            if token == pad:
                                continue
                            token_log_prob = float(top_log_probs_cpu[j])
                            new_tokens = list(beam_state.tokens)
                            ended = False
                            if token == eos:
                                ended = True
                            else:
                                new_tokens.append(token)
                            sequence_candidates[line_number].append(
                                BeamState(
                                    tokens=new_tokens,
                                    log_prob=beam_state.log_prob + token_log_prob,
                                    ended=ended,
                                )
                            )
                            expanded = True

                        if not expanded:
                            # Keep the original beam around so we can continue in the next iteration
                            sequence_candidates[line_number].append(beam_state)

                for line_number, info in list(batch.items()):
                    candidates = sequence_candidates.get(line_number, [])
                    if not candidates:
                        candidates = info.beams

                    candidates.sort(key=lambda beam: beam.log_prob, reverse=True)
                    info.beams = candidates[:beam_size]

                    if all(beam.ended for beam in info.beams):
                        best_beam = info.beams[0]
                        completed_sequences[line_number] = best_beam.tokens[1:]
                        sequences.pop(line_number, None)
                        num_sequences_completed += 1
                        if pbar is not None:
                            pbar.update(1)
                        continue

                    if any((len(beam.tokens) > seq_len) and not beam.ended for beam in info.beams):
                        spillover_sequences[line_number] = info
                        sequences.pop(line_number, None)
                        num_sequences_completed += 1
                        if pbar is not None:
                            pbar.update(1)

            if pbar is not None:
                pbar.close()
            print(f"Total sequences completed: {num_sequences_completed}")
            print(f"Spillover sequences: {len(spillover_sequences)}")

        # We're done with all the buckets. If we still have spillover sequences (or active ones),
        # store the tokens we have generated for those.
        for line_number, info in {**spillover_sequences, **sequences}.items():
            best_beam = max(info.beams, key=lambda beam: beam.log_prob)
            completed_sequences[line_number] = best_beam.tokens[1:]

    # Write all completed sequences to files in order
    with open(output_path_prefix + ".bin", "wb") as output_data_file, open(
        output_path_prefix + ".idx", "wb"
    ) as output_index_file:
        # Get the maximum line number to determine the total number of sequences
        if completed_sequences:
            max_line_number = max(completed_sequences.keys())
            for line_number in range(1, max_line_number + 1):  # line_number starts from 1
                if line_number in completed_sequences:
                    tokens = completed_sequences[line_number]
                else:
                    # Handle missing sequences (should not happen in normal cases)
                    tokens = []
                store_output(output_index_file, output_data_file, tokens)
