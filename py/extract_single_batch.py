#!/usr/bin/env python3
"""
Extract the first batch from the training dataset into a separate single-batch dataset.

This creates a new dataset containing only the entries from the first batch that would
be used by overfit_single_batch.py. The extracted dataset can be used with all existing
tooling (detokenize_dataset, etc.).
"""
import argparse
import os

from batching import EpochBatches
from buckets import open_buckets
from dataset import append_to_dataset, get_entry
from params import target_num_tokens_per_batch, train_dataset_path


def extract_single_batch(
    input_dataset_path: str,
    output_dataset_path: str,
    rng_seed: int = 42,
    target_tokens: int | None = None,
) -> None:
    """
    Extract the first batch from a bucketed dataset into a new dataset.

    Args:
        input_dataset_path: Path to the input dataset (without extension)
        output_dataset_path: Path to the output dataset (without extension)
        rng_seed: Random seed for batch selection (default: 42, matching overfit_single_batch.py)
        target_tokens: Target tokens per batch (default: from params.py)
    """
    if target_tokens is None:
        target_tokens = target_num_tokens_per_batch

    print(f"Opening dataset: {input_dataset_path}")
    print(f"Target tokens per batch: {target_tokens}")
    print(f"Random seed: {rng_seed}")

    # Get the first batch using the same logic as overfit_single_batch.py
    with open_buckets(input_dataset_path) as buckets:
        epoch_batches = EpochBatches(
            num_procs=1,
            proc_id=0,
            bucket_index_file=buckets.bucket_index_file,
            target_num_tokens_per_batch=target_tokens,
            rng_seed=rng_seed,
            full_batches_only=True,
        )

        try:
            batch_id, entry_ids = next(iter(epoch_batches))
        except StopIteration as exc:
            raise RuntimeError("No batches available in the dataset") from exc

        seq_len = (batch_id + 1) * buckets.step_size
        num_entries = len(entry_ids)

        print("\nExtracted batch info:")
        print(f"  Bucket ID: {batch_id}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Number of entries: {num_entries}")
        print(f"  First 10 entry IDs: {entry_ids[:10]}")

        # Create output directory if needed
        output_dir = os.path.dirname(output_dataset_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create the new dataset files
        print(f"\nWriting batch to: {output_dataset_path}")
        with open(f"{output_dataset_path}.src.idx", "wb") as src_idx_file, open(
            f"{output_dataset_path}.src.bin", "wb"
        ) as src_data_file, open(f"{output_dataset_path}.tgt.idx", "wb") as tgt_idx_file, open(
            f"{output_dataset_path}.tgt.bin", "wb"
        ) as tgt_data_file, open(
            f"{output_dataset_path}.meta", "wb"
        ) as meta_file:

            # Create a Dataset wrapper for the output files
            from dataset import Dataset

            output_dataset = Dataset(
                src_idx_file=src_idx_file,
                src_data_file=src_data_file,
                tgt_idx_file=tgt_idx_file,
                tgt_data_file=tgt_data_file,
                meta_file=meta_file,
                src_data_size=0,
                tgt_data_size=0,
                num_entries=0,
            )

            # Copy each entry from the batch to the new dataset
            for i, entry_idx in enumerate(entry_ids):
                corpus_id, original_line_number, src_tokens, tgt_tokens = get_entry(
                    buckets.dataset, entry_idx
                )
                append_to_dataset(
                    output_dataset,
                    corpus_id,
                    original_line_number,
                    src_tokens,
                    tgt_tokens,
                )

                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{num_entries} entries...")

        print(f"\nSuccessfully extracted {num_entries} entries to {output_dataset_path}")
        print("Created files:")
        print(f"  {output_dataset_path}.src.bin")
        print(f"  {output_dataset_path}.src.idx")
        print(f"  {output_dataset_path}.tgt.bin")
        print(f"  {output_dataset_path}.tgt.idx")
        print(f"  {output_dataset_path}.meta")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the first batch from a dataset into a single-batch dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=train_dataset_path,
        help=f"Input dataset path (without extension). Default: {train_dataset_path}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../4_tokens/single_batch",
        help="Output dataset path (without extension). Default: ../4_tokens/single_batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for batch selection (default: 42)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=None,
        help=f"Target tokens per batch (default: {target_num_tokens_per_batch} from params.py)",
    )

    args = parser.parse_args()

    extract_single_batch(
        input_dataset_path=args.input,
        output_dataset_path=args.output,
        rng_seed=args.seed,
        target_tokens=args.target_tokens,
    )


if __name__ == "__main__":
    main()
