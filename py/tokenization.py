import os
import random
import struct
import tempfile
from multiprocessing import Pool, cpu_count

import tokenizers  # type: ignore
from tabulate import tabulate
from tqdm import tqdm

import params
from serialization import append_to_dataset, combine_datasets


def create_tokenizer(output_file_path: str, files: list[str]) -> None:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["[PAD]", "[SOS]", "[EOS]"],
        vocab_size=params.bpe_vocab_size,
        min_frequency=3,
        show_progress=True,
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
    tokenizer.decoder = tokenizers.decoders.Metaspace()
    tokenizer.train(files, trainer)
    tokenizer.save(output_file_path)


def decode(tokenizer: tokenizers.Tokenizer, token: int) -> str:
    if token == 0:
        return "[PAD]"
    elif token == 1:
        return "[SOS]"
    elif token == 2:
        return "[EOS]"
    else:
        return tokenizer.decode([token])  # type: ignore


def pretty_print_tokens(
    tokenizer: tokenizers.Tokenizer, tokens: list[int], chunk_size: int = 20
) -> None:
    """Print tokens in a readable format."""
    rows = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    for tokens_chunk in rows:
        print(tabulate([[decode(tokenizer, token) for token in tokens_chunk], tokens_chunk]))


def test_tokenizer(tokenizer: tokenizers.Tokenizer, text: str) -> None:
    tokens = tokenizer.encode(text).ids
    pretty_print_tokens(tokenizer, tokens)
    print()


def process_line_pair(
    tokenizer: tokenizers.Tokenizer, pair: tuple[str, str]
) -> tuple[list[int], list[int]]:
    src_line, tgt_line = pair
    src_line = src_line.strip()
    tgt_line = tgt_line.strip()
    src_token_obj = tokenizer.encode(src_line)
    tgt_token_obj = tokenizer.encode(tgt_line)
    return src_token_obj.ids, tgt_token_obj.ids


def process_chunk(
    args: tuple[str, str, str, int, int, bool, str, int],
) -> int:
    (
        tokenizer_json_path,
        src_file_path,
        tgt_file_path,
        start_line_idx,
        end_line_idx,
        show_progress_bar,
        tmp_file_path_prefix,
        corpus_id,
    ) = args
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_json_path)

    # Create temporary files for this chunk
    temp_data_file = f"{tmp_file_path_prefix}.bin"
    temp_index_file = f"{tmp_file_path_prefix}.idx"

    # Read only the lines we need from the files
    src_lines = []
    with open(src_file_path, "r", encoding="utf-8") as src_file:
        for i, line in enumerate(src_file):
            if i >= start_line_idx:
                if i >= end_line_idx:
                    break
                src_lines.append(line)

    tgt_lines = []
    with open(tgt_file_path, "r", encoding="utf-8") as tgt_file:
        for i, line in enumerate(tgt_file):
            if i >= start_line_idx:
                if i >= end_line_idx:
                    break
                tgt_lines.append(line)

    chunk = list(zip(src_lines, tgt_lines))

    # Process and write directly to temporary files
    written_entries = 0
    with open(temp_data_file, "wb") as data_file, open(temp_index_file, "wb") as index_file:
        pairs_to_process: list[tuple[str, str]] | tqdm[tuple[str, str]] = chunk
        if show_progress_bar:
            pairs_to_process = tqdm(chunk, desc="Processing chunk")

        for local_i, (src_line, tgt_line) in enumerate(pairs_to_process):
            src_tokens, tgt_tokens = process_line_pair(tokenizer, (src_line, tgt_line))

            if (
                len(src_tokens) == 0
                or len(tgt_tokens) == 0
                or len(src_tokens) > params.max_seq_len
                or (len(tgt_tokens) + 1) > params.max_seq_len
            ):
                continue

            # Convert index to line number
            original_line_number = start_line_idx + local_i + 1

            # Append entry to dataset
            append_to_dataset(
                data_file,
                index_file,
                corpus_id,
                original_line_number,
                src_tokens,
                tgt_tokens,
            )
            written_entries += 1

        if show_progress_bar and hasattr(pairs_to_process, "close"):
            pairs_to_process.close()

    return written_entries


def tokenize_dataset(
    tokenizer_json_path: str,
    output_file_path: str,
    corpus_id: int,
    src_input_file_path: str,
    tgt_input_file_path: str,
) -> None:
    num_cpus = min(16, cpu_count())

    with open(src_input_file_path, "r", encoding="utf-8") as src_input:
        src_line_count = sum(1 for _ in src_input)

    with open(tgt_input_file_path, "r", encoding="utf-8") as tgt_input:
        tgt_line_count = sum(1 for _ in tgt_input)

    assert src_line_count == tgt_line_count, "Length of src and tgt files should be equal"

    print(f"Number of lines: {src_line_count}")
    chunk_size = src_line_count // num_cpus

    # Create temporary directory for chunk files
    temp_dir = tempfile.mkdtemp(prefix="tokenize_")

    # Create tasks with file paths and line ranges instead of actual data
    tasks = []
    file_path_prefixes = []
    for i in range(num_cpus):
        start_line_idx = i * chunk_size
        if i == num_cpus - 1:  # Last chunk gets remaining lines
            end_line_idx = src_line_count
        else:
            end_line_idx = start_line_idx + chunk_size
        tmp_file_path_prefix = os.path.join(temp_dir, f"chunk_{i}")
        file_path_prefixes.append(tmp_file_path_prefix)

        # For simplicity, only show the progress bar for the first chunk
        show_progress_bar = i == 0
        tasks.append(
            (
                tokenizer_json_path,
                src_input_file_path,
                tgt_input_file_path,
                start_line_idx,
                end_line_idx,
                show_progress_bar,
                tmp_file_path_prefix,
                corpus_id,
            )
        )

    with Pool(num_cpus) as pool:
        pool.map(process_chunk, tasks)
        pool.close()  # Prevent any new tasks from being submitted
        pool.join()  # Wait for all worker processes to complete

    print("Combining chunks...")

    # Combine temporary files into final output
    combine_datasets(output_file_path, file_path_prefixes)

    # Remove temporary directory (force)
    for prefix in file_path_prefixes:
        os.unlink(prefix + ".bin")
        os.unlink(prefix + ".idx")
    os.rmdir(temp_dir)


def sample_from_dataset(
    dataset_file_path: str, num_samples: int
) -> list[tuple[int, int, list[int], list[int]]]:
    """
    Sample entries from a binary dataset.

    Args:
        dataset_file_path: Path to the dataset file (without suffix)
        num_samples: Number of samples to return

    Returns:
        List of tuples containing (corpus_id, original_line_number, src_tokens, tgt_tokens)
    """
    data_file_path = dataset_file_path + ".bin"
    index_file_path = dataset_file_path + ".idx"

    # Determine total number of entries by reading index file size
    index_file_size = os.path.getsize(index_file_path)
    total_entries = index_file_size // 5  # Each index entry is now 5 bytes

    if num_samples > total_entries:
        print(f"Requested {num_samples} samples, but dataset only has {total_entries} entries")
        num_samples = total_entries

    # Randomly sample entry indices
    sampled_indices = random.sample(range(total_entries), num_samples)
    sampled_indices.sort()  # Sort for more efficient file access

    results = []

    with open(data_file_path, "rb") as data_file, open(index_file_path, "rb") as index_file:
        for entry_idx in sampled_indices:
            # Read index entry to get data positions
            index_file.seek(entry_idx * 5)  # Each index entry is now 5 bytes
            index_entry = index_file.read(5)
            entry_start_pos, src_token_count = struct.unpack("<IB", index_entry)

            # Read corpus_id and original_line_number
            data_file.seek(entry_start_pos)
            corpus_id = struct.unpack("<B", data_file.read(1))[0]
            original_line_number = struct.unpack("<I", data_file.read(4))[0]

            # Read source tokens (count is stored in index)
            src_tokens = []
            for _ in range(src_token_count):
                token_bytes = data_file.read(2)
                if len(token_bytes) < 2:
                    break
                token = struct.unpack("<H", token_bytes)[0]
                src_tokens.append(token)

            # Target tokens start right after source tokens
            tgt_start_pos = data_file.tell()
            tgt_tokens = []

            # Determine end position (either next entry or end of file)
            if entry_idx + 1 < total_entries:
                # Read next index entry to get end position
                index_file.seek((entry_idx + 1) * 5)  # Each index entry is now 5 bytes
                next_index_entry = index_file.read(5)
                next_entry_start_pos, _ = struct.unpack("<IB", next_index_entry)
                end_pos = next_entry_start_pos
            else:
                # Last entry, read to end of file
                data_file.seek(0, 2)  # Seek to end
                end_pos = data_file.tell()
                data_file.seek(tgt_start_pos)  # Seek back to target position

            # Read target tokens
            while data_file.tell() < end_pos:
                token_bytes = data_file.read(2)
                if len(token_bytes) < 2:
                    break
                token = struct.unpack("<H", token_bytes)[0]
                tgt_tokens.append(token)

            results.append((corpus_id, original_line_number, src_tokens, tgt_tokens))

    return results
