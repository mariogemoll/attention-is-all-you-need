import os

import tokenizers  # type: ignore

from dataset_test_helpers import make_sequential_single_stream_dataset
from tokenization import detokenize_dataset


def test_detokenize_dataset_end_to_end(tmp_path: str) -> None:
    # Create a tiny tokenizer mapping ids -> words
    tok = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocab={"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "hello": 5, "world": 6, "foo": 7},
            unk_token="[UNK]",
        )
    )
    tok_path = os.path.join(tmp_path, "tok.json")
    tok.save(tok_path)

    # Build a small simple-format dataset
    ds_prefix = os.path.join(tmp_path, "ds")
    # Entries: [hello world EOS], [SOS hello EOS], [EOS]
    entries = [
        [5, 6, 2],  # "hello world"
        [1, 5, 2],  # "hello"
        [2],  # empty
    ]
    make_sequential_single_stream_dataset(ds_prefix, entries)

    out_txt = os.path.join(tmp_path, "out.txt")
    detokenize_dataset(tok_path, ds_prefix, out_txt)

    with open(out_txt, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    assert lines == ["hello world", "hello", ""]
