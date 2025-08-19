import sacrebleu


def get_bleu_score(reference_file_path: str, translation_file_path: str) -> float:
    bleu = sacrebleu.BLEU()
    with open(reference_file_path, "r") as reference_file, open(
        translation_file_path, "r"
    ) as translation_file:
        reference_lines = reference_file.readlines()
        translation_lines = translation_file.readlines()

        assert len(reference_lines) == len(
            translation_lines
        ), "Input and output files must have the same number of lines."

        result = bleu.corpus_score(translation_lines, [reference_lines])
        score: float = result.score
        return score
