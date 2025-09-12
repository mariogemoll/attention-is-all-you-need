import os
import random
import struct
import tempfile
from multiprocessing import Pool, cpu_count
from typing import Union

import tokenizers  # type: ignore
from tabulate import tabulate
from tqdm import tqdm

from data import from_uint16_le_bytes
from indexed_out_of_order import add_entry as add_out_of_order_entry
from indexed_out_of_order import convert_to_sequential
from indexed_sequential import append_indexed_entry, read_indexed_entry
from params import max_parallelism, max_seq_len, vocab_size


def create_tokenizer(output_file_path: str, files: list[str]) -> None:
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    trainer = tokenizers.trainers.BpeTrainer(
        special_tokens=["[PAD]", "[SOS]", "[EOS]"],
        vocab_size=vocab_size,
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


def detokenize_sequence(tokenizer: tokenizers.Tokenizer, tokens: list[int]) -> str:
    """
    Convert a list of tokens back to text.

    Args:
        tokenizer: The tokenizer to use for decoding
        tokens: List of token IDs

    Returns:
        Detokenized text string
    """
    if not tokens:
        return ""

    # Remove special tokens and decode the rest
    text_tokens = []
    for token in tokens:
        if token == 0:  # PAD
            continue
        elif token == 1:  # SOS
            continue
        elif token == 2:  # EOS
            break
        else:
            text_tokens.append(token)

    if not text_tokens:
        return ""

    return str(tokenizer.decode(text_tokens))


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
    args: tuple[str, str, str, int, int, bool, str, int, str],
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
        tmp_meta_file_path,
    ) = args
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_json_path)

    temp_src_bin = f"{tmp_file_path_prefix}.src.bin"
    temp_src_idx = f"{tmp_file_path_prefix}.src.idx"
    temp_tgt_bin = f"{tmp_file_path_prefix}.tgt.bin"
    temp_tgt_idx = f"{tmp_file_path_prefix}.tgt.idx"
    temp_meta = tmp_meta_file_path

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
    written_entries = 0
    src_pos = 0
    tgt_pos = 0
    with open(temp_src_bin, "wb") as src_bin, open(temp_src_idx, "wb") as src_idx, open(
        temp_tgt_bin, "wb"
    ) as tgt_bin, open(temp_tgt_idx, "wb") as tgt_idx, open(temp_meta, "wb") as meta_file:
        pairs_to_process: list[tuple[str, str]] | tqdm[tuple[str, str]] = chunk
        if show_progress_bar:
            pairs_to_process = tqdm(chunk, desc="Processing chunk")

        for local_i, (src_line, tgt_line) in enumerate(pairs_to_process):
            src_tokens, tgt_tokens = process_line_pair(tokenizer, (src_line, tgt_line))

            if (
                len(src_tokens) == 0
                or len(tgt_tokens) == 0
                or len(src_tokens) > max_seq_len
                or (len(tgt_tokens) + 1) > max_seq_len
            ):
                continue

            original_line_number = start_line_idx + local_i + 1

            # Write src tokens
            src_bin.write(struct.pack(f"<{len(src_tokens)}H", *src_tokens))
            src_pos += len(src_tokens) * 2
            src_idx.write(struct.pack("<I", src_pos))

            # Write tgt tokens
            tgt_bin.write(struct.pack(f"<{len(tgt_tokens)}H", *tgt_tokens))
            tgt_pos += len(tgt_tokens) * 2
            tgt_idx.write(struct.pack("<I", tgt_pos))

            # Write metadata: corpus_id (1B), original_line_number (4B)
            meta_file.write(struct.pack("<BI", corpus_id, original_line_number))

            written_entries += 1

        # tqdm objects have .close(), lists do not. Only call if it's tqdm
        if show_progress_bar and isinstance(pairs_to_process, tqdm):
            pairs_to_process.close()

    return written_entries


def tokenize_dataset(
    tokenizer_json_path: str,
    output_file_path_prefix: str,
    corpus_id: int,
    src_input_file_path: str,
    tgt_input_file_path: str,
) -> None:
    num_cpus = min(cpu_count(), max_parallelism)

    with open(src_input_file_path, "r", encoding="utf-8") as src_input:
        src_line_count = sum(1 for _ in src_input)

    with open(tgt_input_file_path, "r", encoding="utf-8") as tgt_input:
        tgt_line_count = sum(1 for _ in tgt_input)

    assert src_line_count == tgt_line_count, "Length of src and tgt files should be equal"

    print(f"Number of lines: {src_line_count}")
    chunk_size = src_line_count // num_cpus

    temp_dir = tempfile.mkdtemp(prefix="tokenize_")
    tasks = []
    file_path_prefixes = []
    meta_file_paths = []
    for i in range(num_cpus):
        start_line_idx = i * chunk_size
        if i == num_cpus - 1:
            end_line_idx = src_line_count
        else:
            end_line_idx = start_line_idx + chunk_size
        tmp_file_path_prefix = os.path.join(temp_dir, f"chunk_{i}")
        tmp_meta_file_path = os.path.join(temp_dir, f"chunk_{i}.meta")
        file_path_prefixes.append(tmp_file_path_prefix)
        meta_file_paths.append(tmp_meta_file_path)
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
                tmp_meta_file_path,
            )
        )

    with Pool(num_cpus) as pool:
        pool.map(process_chunk, tasks)
        pool.close()
        pool.join()

    print("Combining chunks...")

    # Combine src and tgt bin/idx files
    def combine_bin_idx(bin_or_idx: str, out_prefix: str, chunk_prefixes: list[str]) -> None:
        out_path = f"{out_prefix}.{bin_or_idx}"
        with open(out_path, "wb") as out_f:
            offset = 0
            for prefix in chunk_prefixes:
                in_path = f"{prefix}.{bin_or_idx}"
                with open(in_path, "rb") as in_f:
                    if bin_or_idx.endswith("bin"):
                        # Just copy
                        while True:
                            chunk = in_f.read(65536)
                            if not chunk:
                                break
                            out_f.write(chunk)
                    else:
                        # idx: need to adjust offsets
                        while True:
                            entry = in_f.read(4)
                            if not entry:
                                break
                            end_pos = struct.unpack("<I", entry)[0]
                            out_f.write(struct.pack("<I", end_pos + offset))
                        # Update offset for next chunk
                        if bin_or_idx.endswith("bin"):
                            in_f.seek(0, 2)
                            offset += in_f.tell()
                        else:
                            # For idx, get last end_pos
                            in_f.seek(-4, 2)
                            last_end = struct.unpack("<I", in_f.read(4))[0]
                            offset += last_end

    combine_bin_idx("src.bin", output_file_path_prefix, file_path_prefixes)
    combine_bin_idx("src.idx", output_file_path_prefix, file_path_prefixes)
    combine_bin_idx("tgt.bin", output_file_path_prefix, file_path_prefixes)
    combine_bin_idx("tgt.idx", output_file_path_prefix, file_path_prefixes)

    # Combine meta files
    out_meta_path = f"{output_file_path_prefix}.meta"
    with open(out_meta_path, "wb") as out_meta:
        for meta_path in meta_file_paths:
            with open(meta_path, "rb") as in_meta:
                while True:
                    chunk = in_meta.read(65536)
                    if not chunk:
                        break
                    out_meta.write(chunk)

    # Remove temporary directory and files
    for prefix in file_path_prefixes:
        for ext in ["src.bin", "src.idx", "tgt.bin", "tgt.idx"]:
            try:
                os.unlink(f"{prefix}.{ext}")
            except Exception:
                pass
    for meta_path in meta_file_paths:
        try:
            os.unlink(meta_path)
        except Exception:
            pass
    os.rmdir(temp_dir)


def _detokenize_process_chunk(args: tuple[str, str, str, int, int, str, bool]) -> int:
    (
        tokenizer_path,
        input_data_path,
        input_index_path,
        start_idx,
        end_idx,
        out_prefix,
        show_progress,
    ) = args
    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    out_data_path = f"{out_prefix}.bin"
    out_index_path = f"{out_prefix}.idx"
    written = 0
    with open(input_data_path, "rb") as data_file, open(input_index_path, "rb") as index_file, open(
        out_data_path, "wb"
    ) as out_data, open(out_index_path, "wb") as out_index:
        base_iterator = range(start_idx, end_idx)
        iterator: Union[range, tqdm[int]] = base_iterator
        if show_progress:
            iterator = tqdm(base_iterator, desc=f"Detok chunk {os.path.basename(out_prefix)}")
        for i in iterator:
            try:
                record_bytes = read_indexed_entry(data_file, index_file, i)
                tokens = from_uint16_le_bytes(record_bytes)
                text = detokenize_sequence(tokenizer, tokens)
                if "\n" in text:
                    raise ValueError("Detokenized text contains a newline character")
                entry_bytes = struct.pack("<I", i) + text.encode("utf-8")
            except Exception:
                # On error, write empty entry with original index
                entry_bytes = struct.pack("<I", i) + b""
            append_indexed_entry(out_data, out_index, entry_bytes)
            written += 1
    return written


def detokenize_dataset(
    tokenizer_path: str,
    input_dataset_path: str,
    output_file_path: str,
) -> None:
    """
    Parallel detokenization using sequential + out_of_order indices for ordering.

    Workers write sequential files with entry payload = ORIGINAL_INDEX(4B) + UTF-8 text.
    Main merges into an out_of_order index using ORIGINAL_INDEX, then converts to sequential
    and finally writes the ordered text file.
    """
    # Input files
    input_data_path = input_dataset_path + ".bin"
    input_index_path = input_dataset_path + ".idx"

    if not os.path.exists(input_data_path) or not os.path.exists(input_index_path):
        raise FileNotFoundError(
            f"Error: Input dataset files not found: {input_dataset_path}.bin/.idx"
        )

    # Entry count from sequential format (4 bytes per index entry)
    input_index_size = os.path.getsize(input_index_path)
    num_entries = input_index_size // 4

    print(f"Detokenizing {num_entries:,} entries from {input_dataset_path}")

    # Prepare temp workspace
    temp_dir = tempfile.mkdtemp(prefix="detok_")
    try:
        num_cpus = min(cpu_count(), max_parallelism)
        chunk_size = max(1, (num_entries + num_cpus - 1) // num_cpus)
        tasks: list[tuple[str, str, str, int, int, str, bool]] = []
        prefixes: list[str] = []
        for i in range(num_cpus):
            start = i * chunk_size
            if start >= num_entries:
                break
            end = min(start + chunk_size, num_entries)
            out_prefix = os.path.join(temp_dir, f"chunk_{i}")
            prefixes.append(out_prefix)
            show_progress = i == 0
            tasks.append(
                (
                    tokenizer_path,
                    input_data_path,
                    input_index_path,
                    start,
                    end,
                    out_prefix,
                    show_progress,
                )
            )

        # Process chunks in parallel
        if len(tasks) == 1:
            _detokenize_process_chunk(tasks[0])
        else:
            with Pool(len(tasks)) as pool:
                list(pool.map(_detokenize_process_chunk, tasks))

        # Merge into out_of_order dataset using original indices
        ooo_index_path = os.path.join(temp_dir, "combined.ooo.idx")
        ooo_data_path = os.path.join(temp_dir, "combined.ooo.bin")
        with open(ooo_index_path, "wb+") as ooo_idx, open(ooo_data_path, "wb+") as ooo_bin:
            for prefix in prefixes:
                seq_idx_path = f"{prefix}.idx"
                seq_bin_path = f"{prefix}.bin"
                with open(seq_idx_path, "rb") as sidx, open(seq_bin_path, "rb") as sbin:
                    sidx.seek(0, 2)
                    n = sidx.tell() // 4
                    sidx.seek(0)
                    for j in range(n):
                        # Read the record bytes via sequential index
                        record = read_indexed_entry(sbin, sidx, j)
                        if len(record) < 4:
                            continue
                        orig_idx = struct.unpack("<I", record[:4])[0]
                        payload = record[4:]
                        add_out_of_order_entry(ooo_idx, ooo_bin, orig_idx, payload)

        # Convert to final sequential dataset
        seq_index_path = os.path.join(temp_dir, "final.idx")
        seq_data_path = os.path.join(temp_dir, "final.bin")
        convert_to_sequential(ooo_index_path, ooo_data_path, seq_index_path, seq_data_path)

        # Write ordered text output
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        empty_count = 0
        with open(seq_data_path, "rb") as sbin, open(seq_index_path, "rb") as sidx, open(
            output_file_path, "w", encoding="utf-8"
        ) as out_txt:
            sidx.seek(0, 2)
            n = sidx.tell() // 4
            sidx.seek(0)
            for i in tqdm(range(n), desc="Writing output"):
                data = read_indexed_entry(sbin, sidx, i)
                try:
                    text = data.decode("utf-8")
                    if "\n" in text:
                        raise ValueError("Detokenized text contains a newline character")
                except Exception:
                    text = ""
                if not text.strip():
                    empty_count += 1
                out_txt.write(text + "\n")

        print("✓ Detokenization complete!")
        print(f"  Total entries processed: {num_entries:,}")
        print(f"  Output file: {output_file_path}")
        print(f"  Empty sequences: {empty_count:,}")
    finally:
        # Cleanup temp files
        try:
            for prefix in prefixes if "prefixes" in locals() else []:
                for ext in (".bin", ".idx"):
                    p = f"{prefix}{ext}"
                    if os.path.exists(p):
                        os.unlink(p)
            for p in [
                os.path.join(temp_dir, "combined.ooo.idx"),
                os.path.join(temp_dir, "combined.ooo.bin"),
                os.path.join(temp_dir, "final.idx"),
                os.path.join(temp_dir, "final.bin"),
            ]:
                if os.path.exists(p):
                    os.unlink(p)
            os.rmdir(temp_dir)
        except Exception:
            pass


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

    total_entries = get_number_of_entries(dataset_file_path)

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


def get_number_of_entries(dataset_file_path: str) -> int:
    """
    Return the number of entries in a dataset that uses a 5-byte index entry
    format: 4-byte start offset (uint32 LE) + 1-byte source-token-count.

    This helper reads the `.idx` file size and divides by 5.
    """
    index_file_path = dataset_file_path + ".idx"
    size = os.path.getsize(index_file_path)
    return size // 5
