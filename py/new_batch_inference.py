import argparse
import sys
import tempfile
from typing import BinaryIO, Iterator

import torch
from tqdm import tqdm

import params
from buckets import (
    BucketedDataset,
    get_bucket_size,
    get_bucket_sizes,
    get_entry_idx_from_bucket,
    open_buckets,
)
from data import to_uint16_le_bytes
from dataset import get_entry
from indexed_out_of_order import add_entry, convert_to_sequential
from model import Transformer
from util import get_device


def calculate_adaptive_batch_size(seq_len: int, target_tokens: int, alignment: int) -> int:
    """
    Calculate batch size based on sequence length to maintain roughly constant token count.

    Args:
        seq_len: Maximum sequence length for this bucket
        target_tokens: Target total tokens per batch
        alignment: Batch size must be multiple of this value

    Returns:
        Batch size aligned to the specified alignment
    """
    # Calculate ideal batch size for target token count
    ideal_batch_size = target_tokens // seq_len

    # Ensure minimum batch size of alignment
    if ideal_batch_size < alignment:
        ideal_batch_size = alignment

    # Round to nearest multiple of alignment
    aligned_batch_size = ((ideal_batch_size + alignment - 1) // alignment) * alignment

    return aligned_batch_size


class TranslationItem:
    id: int  # Index of the sequence in the dataset
    src_tokens: list[int]
    tgt_tokens: list[int]

    src_encoding: torch.Tensor
    hypotheses: list[tuple[list[int], float]]  # List of (token list, score)


class BucketBatchIterator:
    """
    A class that provides batch iteration over a specific bucket with len() support.
    """

    def __init__(
        self,
        dataset: BucketedDataset,
        bucket_id: int,
        model: Transformer,
        device: torch.device,
        target_tokens: int = 8192,
        batch_alignment: int = 16,
    ) -> None:
        self.dataset = dataset
        self.bucket_id = bucket_id
        self.model = model
        self.device = device
        self.max_seq_len = (bucket_id + 1) * dataset.step_size

        # Calculate adaptive batch size based on sequence length
        self.batch_size = calculate_adaptive_batch_size(
            self.max_seq_len, target_tokens, batch_alignment
        )

        # Get bucket info
        bucket_sizes = get_bucket_sizes(dataset.bucket_index_file)

        if bucket_id >= len(bucket_sizes):
            self.bucket_size = 0
        else:
            self.bucket_size = bucket_sizes[bucket_id]

        # Calculate number of batches using the adaptive batch size
        self._num_batches = (
            (self.bucket_size + self.batch_size - 1) // self.batch_size
            if self.bucket_size > 0
            else 0
        )

    def __len__(self) -> int:
        """Return the number of batches in this bucket."""
        return self._num_batches

    def __iter__(self) -> Iterator[list[TranslationItem]]:
        """Iterate over batches of encoded TranslationItems."""
        if self.bucket_size == 0:
            return

        # Process bucket in batches
        for start_idx in range(0, self.bucket_size, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.bucket_size)
            current_batch_size = end_idx - start_idx

            # Collect batch items
            batch_items = []
            for idx_in_bucket in range(start_idx, end_idx):
                entry_idx = get_entry_idx_from_bucket(
                    self.dataset.bucket_index_file, self.bucket_id, idx_in_bucket
                )
                corpus_id, original_line_number, src_tokens, tgt_tokens = get_entry(
                    self.dataset.dataset, entry_idx
                )

                item = TranslationItem()
                item.id = entry_idx  # Use entry_idx as the sequence id
                item.src_tokens = src_tokens
                item.tgt_tokens = tgt_tokens
                item.hypotheses = [([params.sos], 0.0)]

                batch_items.append(item)

            # Create batch tensor for encoding
            batch_src = torch.full(
                (current_batch_size, self.max_seq_len),
                params.pad,
                dtype=torch.long,
                device=self.device,
            )

            # Fill batch tensor
            for i, item in enumerate(batch_items):
                seq_len = len(item.src_tokens)
                batch_src[i, :seq_len] = torch.tensor(
                    item.src_tokens, dtype=torch.long, device=self.device
                )

            # Batch encode
            with torch.no_grad():
                batch_encodings = self.model.encode(batch_src)  # (batch_size, max_seq_len, d_model)

            # Store encodings in items (trim padding)
            for i, item in enumerate(batch_items):
                seq_len = len(item.src_tokens)
                item.src_encoding = batch_encodings[i, :seq_len]  # (seq_len, d_model)

            # print(f"    ✓ Encoded batch shape: {batch_encodings.shape}")

            yield batch_items


def iter_encoded_batches(
    dataset: BucketedDataset,
    bucket_id: int,
    model: Transformer,
    device: torch.device,
    target_tokens: int = 8192,
    batch_alignment: int = 16,
) -> BucketBatchIterator:
    """
    Iterator that yields batches of encoded TranslationItems for a specific bucket.

    Args:
        dataset: BucketedDataset instance
        bucket_id: ID of the bucket to process
        model: Transformer model for encoding
        device: torch device
        target_tokens: Target total tokens per batch
        batch_alignment: Batch size alignment (must be multiple of this)

    Yields:
        List of TranslationItems with encodings (one batch at a time)
    """
    return BucketBatchIterator(dataset, bucket_id, model, device, target_tokens, batch_alignment)


def process_bucket(
    device: torch.device,
    model: Transformer,
    dataset: BucketedDataset,
    ooo_index: BinaryIO,
    ooo_data: BinaryIO,
    bucket_id: int,
    target_tokens: int,
    batch_alignment: int,
) -> list[TranslationItem]:
    bucket_size = get_bucket_size(dataset.bucket_index_file, bucket_id)
    completed_count = 0
    spillover_items: list[TranslationItem] = []
    bucket_seq_len = (bucket_id + 1) * dataset.step_size
    items: dict[int, TranslationItem] = {}

    bucket_iterator = BucketBatchIterator(
        dataset, bucket_id, model, device, target_tokens, batch_alignment
    )
    adaptive_batch_size = bucket_iterator.batch_size
    iterator = iter(bucket_iterator)  # Create iterator ONCE

    # Create progress bar for this bucket
    pbar = tqdm(total=bucket_size, desc=f"Bucket {bucket_id}", unit="sequences")

    # Pre-allocate tensors outside the loop for reuse
    enc_input = torch.full(
        (adaptive_batch_size, bucket_seq_len), params.pad, dtype=torch.long, device=device
    )
    memory = torch.empty(
        (adaptive_batch_size, bucket_seq_len, params.num_model_dims),
        dtype=torch.float,
        device=device,
    )
    dec_input = torch.full(
        (adaptive_batch_size, bucket_seq_len), params.pad, dtype=torch.long, device=device
    )

    while completed_count + len(spillover_items) < bucket_size:
        # Update progress bar
        completed = completed_count + len(spillover_items)
        pbar.n = completed
        pbar.set_postfix(
            {"completed": completed_count, "spillover": len(spillover_items), "active": len(items)}
        )
        pbar.refresh()

        # Safety check: if no items are being processed and no new items can be loaded, break
        prev_total = len(items)

        while len(items) < adaptive_batch_size:
            try:
                batch_items = next(iterator)  # Use the same iterator
                for item in batch_items:
                    items[item.id] = item
            except StopIteration:
                # No more items in this bucket
                break

        # Safety check: if no progress was made (no new items loaded, no items processed), break to
        # avoid infinite loop
        if len(items) == prev_total and len(items) == 0:
            print(
                "Warning: No items to process and no new items available. Breaking from bucket "
                f"{bucket_id}."
            )
            break

        # Clear/reset reused tensors instead of reallocating
        enc_input.fill_(params.pad)
        dec_input.fill_(params.pad)
        # Note: memory tensor will be filled with encoder outputs, so no need to clear

        # Generate an iterator over items values
        item_values = list(items.values())
        num_entries_in_batch = 0
        metadata: list[tuple[int, int]] = []  # List of (id, hypothesis_idx)

        new_hypotheses: dict[int, list[tuple[list[int], float]]] = {}

        for item in item_values:
            if num_entries_in_batch >= adaptive_batch_size:
                break
            # Get the number of not-finished hypotheses
            num_not_finished = 0
            for h, _ in item.hypotheses:
                if h[-1] != params.eos:
                    num_not_finished += 1

            if num_entries_in_batch + num_not_finished <= adaptive_batch_size:
                for i, (h, prob) in enumerate(item.hypotheses):
                    if h[-1] == params.eos:
                        if item.id not in new_hypotheses:
                            new_hypotheses[item.id] = []
                        new_hypotheses[item.id].append((h, prob))
                    else:
                        # Copy source tokens to encoder input
                        seq_len = len(item.src_tokens)
                        enc_input[num_entries_in_batch, :seq_len] = torch.tensor(
                            item.src_tokens, dtype=torch.long, device=device
                        )
                        # Copy source encoding to memory
                        memory[num_entries_in_batch, :seq_len, :] = item.src_encoding
                        # Copy hypothesis tokens to decoder input
                        hyp_len = len(h)
                        dec_input[num_entries_in_batch, :hyp_len] = torch.tensor(
                            h, dtype=torch.long, device=device
                        )
                        metadata.append((item.id, i))
                        num_entries_in_batch += 1

        # Decode
        with torch.no_grad():
            # Generate output logits
            # print(f"Decoding batch with {num_entries_in_batch} entries...")
            output_logits = model.decode(enc_input, memory, dec_input)

            for i in range(len(metadata)):
                item_id, hypothesis_idx = metadata[i]
                item = items[item_id]
                # Get the length of the sequence
                seq_len = len(item.hypotheses[hypothesis_idx][0])
                current_hypothesis = item.hypotheses[hypothesis_idx]

                if item_id not in new_hypotheses:
                    new_hypotheses[item_id] = []

                # Add the current hypothesis plus each of the topk tokens
                # Convert logits to log probabilities for proper scoring
                position_logits = output_logits[i, seq_len - 1]
                log_probs = torch.log_softmax(position_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(log_probs, k=3)

                for log_prob, idx in zip(topk_log_probs.tolist(), topk_indices.tolist()):
                    new_score = (
                        current_hypothesis[1] + log_prob
                    )  # Adding log probabilities (CORRECT)
                    new_tokens = current_hypothesis[0] + [idx]
                    new_hyp = (new_tokens, new_score)
                    new_hypotheses[item_id].append(new_hyp)

            # Keep the best 3 hypotheses per item
            for key, hyps in new_hypotheses.items():
                hyps.sort(key=lambda x: x[1], reverse=True)
                items[key].hypotheses = hyps[:3]

            # If we have 3 finished hyptheses for an item, we can consider it done
            done_items = [
                key
                for key, item in items.items()
                if len(item.hypotheses) >= 3
                and all(h[0][-1] == params.eos for h in item.hypotheses)
            ]
            for key in done_items:
                # Get the best hypothesis
                best_hypothesis = items[key].hypotheses[0]
                # Write tokens as 16bit le integers
                add_entry(ooo_index, ooo_data, key, to_uint16_le_bytes(best_hypothesis[0]))
                del items[key]
                completed_count += 1

            # If any of the hypotheses exceeds the max length, we move the item to spillover
            spillover_keys = []
            for key, item in items.items():
                if any(len(h[0]) >= bucket_seq_len for h in item.hypotheses):
                    spillover_keys.append(key)

            for key in spillover_keys:
                spillover_items.append(items[key])
                del items[key]

    # Close progress bar
    pbar.close()
    return spillover_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket-based batch translation")
    parser.add_argument(
        "model_weights_file",
        help="Path to model weights file (e.g., model_0007.pt)",
    )
    parser.add_argument(
        "input_dataset_prefix",
        help="Path prefix to bucketed dataset (e.g., 'data/test' for data/test.bidx)",
    )
    parser.add_argument(
        "output_prefix",
        help="Path to output for translations",
    )
    parser.add_argument(
        "--beam-search", action="store_true", help="Use beam search instead of greedy decoding"
    )
    parser.add_argument(
        "--beam-size", type=int, default=4, help="Beam size for beam search (default: 4)"
    )
    parser.add_argument(
        "--num-entries", type=int, default=10, help="Number of entries to process (default: 10)"
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=8192,
        help="Target token count per batch (default: 8192)",
    )
    parser.add_argument(
        "--batch-alignment", type=int, default=16, help="Batch size alignment (default: 16)"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_weights_file}...")
    device = get_device()
    print(f"Using device: {device}")
    model = Transformer().to(device)

    try:
        model_state_dict = torch.load(args.model_weights_file, map_location=device)
        model.load_state_dict(model_state_dict)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Open bucketed dataset
    print(f"Opening bucketed dataset from {args.input_dataset_prefix}...")

    output_idx_path = args.output_prefix + ".idx"
    output_data_path = args.output_prefix + ".bin"
    with open_buckets(args.input_dataset_prefix) as dataset, tempfile.TemporaryFile(
        "w+b"
    ) as ooo_index, tempfile.TemporaryFile("w+b") as ooo_data:
        print(f"Found {dataset.num_buckets} buckets with step_size={dataset.step_size}")

        # Process buckets

        for bucket_id in range(dataset.num_buckets):
            process_bucket(
                device,
                model,
                dataset,
                ooo_index,
                ooo_data,
                bucket_id,
                args.target_tokens,
                args.batch_alignment,
            )

        # Seek to beginning of temp files for reading
        ooo_index.seek(0)
        ooo_data.seek(0)

        with open(output_idx_path, "wb") as seq_idx, open(output_data_path, "wb") as seq_data:
            convert_to_sequential(ooo_index, ooo_data, seq_idx, seq_data)

        print("\n✓ Finished processing all buckets")


if __name__ == "__main__":
    main()
