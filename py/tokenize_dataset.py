import sys

from tokenization import tokenize_dataset

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print(
            f"Usage: python {sys.argv[0]} tokenizer_json output_file corpus_id "
            + "src_input_file tgt_input_file"
        )
        sys.exit(1)

    tokenizer_json_path = sys.argv[1]
    output_file_path = sys.argv[2]
    corpus_id = int(sys.argv[3])
    src_input_file_path = sys.argv[4]
    tgt_input_file_path = sys.argv[5]

    tokenize_dataset(
        tokenizer_json_path, output_file_path, corpus_id, src_input_file_path, tgt_input_file_path
    )
