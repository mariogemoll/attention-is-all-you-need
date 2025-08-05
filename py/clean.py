import multiprocessing
import os
import random
import string
import sys
import unicodedata
from collections import Counter

from tqdm import tqdm, trange

standard_alphabet = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

interpunctuation = set(".,:;!?\"'()-/")

digits = set("0123456789")

latin_extended_a = (
    "ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘř"
    "ŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ"
)

latin_1_supplement = "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"

other = set(" +=*#&€£$¥%@§°®<>_{}")

allowed_chars = (
    standard_alphabet.union(interpunctuation)
    .union(digits)
    .union(latin_extended_a)
    .union(latin_1_supplement)
    .union(other)
)


def merge_files(temp_files: list[str], output_file: str) -> None:
    """Merge temporary files into one output file."""
    with open(output_file, "w") as outfile:
        for fname in temp_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            os.remove(fname)  # Clean up temporary file


def process_line(line: bytes) -> tuple[str, set[str]]:
    if line == b"":
        return "", set()
    text = unicodedata.normalize("NFKC", line.decode("utf-8"))
    text = text.replace("\u00ad", "")  # Remove soft hyphen
    text = text.replace("\u200b", "")  # Remove zero-width space
    text = text.replace("\u0082", "")  # Remove break permitted here
    text = text.replace("\ufeff", "")  # Remove zero-width no-break space
    text = text.replace("|", "/")  # Replace pipe with solidus
    text = text.replace("⁄", "/")  # Replace fraction slash with solidus
    text = text.replace("×", "x")  # Replace multiplication sign with x
    text = text.replace("[", "(")  # Replace opening square bracket with opening parenthesis
    text = text.replace("]", ")")  # Replace closing square bracket with closing parenthesis
    for quotation_mark in "›‹»«“”‟„":
        text = text.replace(quotation_mark, '"')
    for hyphen_like in "\u2010\u2013\u2014\u2015\u2212":
        text = text.replace(hyphen_like, "-")
    for apostrophe_like in "’‘′`‚":
        text = text.replace(apostrophe_like, "'")
    chars_in_line = set(text)
    unknown_chars = chars_in_line - allowed_chars
    if unknown_chars:
        text = ""
    return text, unknown_chars


def process_chunk(args: tuple[int, int, int, str, str, str]) -> tuple[str, str, Counter[str]]:
    """Process a chunk of lines and return the temporary file names."""
    start_line, end_line, worker_id, src_input_path, tgt_input_path, run_id = args

    src_tmp_file = f"tmp_{run_id}_{worker_id}_src.txt"
    tgt_tmp_file = f"tmp_{run_id}_{worker_id}_tgt.txt"

    unknown_chars_counter: Counter[str] = Counter()
    with (
        open(src_input_path, "rb") as src_input,
        open(tgt_input_path, "rb") as tgt_input,
        open(src_tmp_file, "w") as src_output,
        open(tgt_tmp_file, "w") as tgt_output,
    ):
        # Skip to the start line
        for i in range(start_line):
            src_input.readline()
            tgt_input.readline()

        processed_count = 0
        progress_iterator: tqdm[int] | range
        if worker_id == 0:
            progress_iterator = trange(start_line, end_line, ncols=100, file=sys.stdout)
        else:
            progress_iterator = range(start_line, end_line)

        for i in progress_iterator:
            src_line = src_input.readline()
            tgt_line = tgt_input.readline()
            if not src_line and not tgt_line:
                # Reached the end of the file
                break

            if src_line.endswith(b"\n"):
                src_line = src_line[:-1]
            if tgt_line.endswith(b"\n"):
                tgt_line = tgt_line[:-1]

            processed_src_line, unknown_chars_src = process_line(src_line)
            processed_tgt_line, unknown_chars_tgt = process_line(tgt_line)
            if processed_src_line == "" or processed_tgt_line == "":
                processed_src_line = processed_tgt_line = ""
            unknown_chars_pair = unknown_chars_src.union(unknown_chars_tgt)
            if unknown_chars_pair:
                unknown_chars_counter.update(unknown_chars_pair)
            src_output.write(processed_src_line + "\n")
            tgt_output.write(processed_tgt_line + "\n")
            processed_count += 1

    return src_tmp_file, tgt_tmp_file, unknown_chars_counter


def count_lines(file_path: str) -> int:
    """Count the number of lines in a file."""
    with open(file_path, "rb") as f:
        count = 0
        for line in f:
            count += 1
    return count


def generate_random_string(length: int = 8) -> str:
    """Generate a random string of fixed length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def clean(
    src_input: str,
    tgt_input: str,
    src_output: str,
    tgt_output: str,
    num_processes: int,
    print_unknown_chars: bool = False,
) -> None:

    total_lines = count_lines(src_input)

    # Calculate chunk size for each worker
    chunk_size = total_lines // num_processes
    remainder = total_lines % num_processes

    # Prepare arguments for each worker
    run_id = generate_random_string()
    worker_args = []

    current_start = 0
    for i in range(num_processes):
        # Add one extra line to the first 'remainder' workers
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        current_end = current_start + current_chunk_size

        worker_args.append((current_start, current_end, i, src_input, tgt_input, run_id))
        current_start = current_end

    # Use multiprocessing Pool to process chunks
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, worker_args)

    if print_unknown_chars:
        unknown_chars_counter: Counter[str] = Counter()
        for _, _, unknown_chars in results:
            unknown_chars_counter.update(unknown_chars)

        if not unknown_chars_counter:
            print("No unknown characters found.")
        else:
            # Print the most common unknown characters
            print("Most common unknown characters:")
            most_common = unknown_chars_counter.most_common(20)
            # Get the number of digits of the largest count
            max_count_length = len(str(max(count for _, count in most_common)))

            for char, count in most_common:
                # Print count(right-aligned), character, Unicode code point and character name
                name = unicodedata.name(char, "UNKNOWN")
                print(f"{count:>{max_count_length}} {char} U+{ord(char):04X} {name}")

    # Separate source and target temporary files
    src_tmp_files = []
    tgt_tmp_files = []

    for [src_tmp_file, tgt_tmp_file, _] in results:
        if src_tmp_file and tgt_tmp_file:  # Check for successful processing
            src_tmp_files.append(src_tmp_file)
            tgt_tmp_files.append(tgt_tmp_file)

    # Merge all temporary files into the final output file
    merge_files(src_tmp_files, src_output)
    merge_files(tgt_tmp_files, tgt_output)


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <src_input> <tgt_input> <src_output> <tgt_output>")
        sys.exit(1)

    num_processes = multiprocessing.cpu_count()
    # num_processes = 1
    clean(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], num_processes)
