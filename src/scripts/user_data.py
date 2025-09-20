from pathlib import Path
import json
import pprint

import sys

from ..induction import CsarPipeline
from ..util import load_input_data, print_morphemes


def main(
    input_path: Path,
    *,
    max_lines: int | None,
    max_inventory_size: int | None,
) -> None:
    data = load_input_data(input_path)
    dataset = data[:max_lines]
    pipeline = CsarPipeline(
        dataset,
        max_ngram=999_999,
        max_semcomps=999_999,
        max_inventory_size=max_inventory_size,
        trim_threshold=0,
        search_best_sub=True,
        vocab_size=None,
        token_vocab_size=None,
        ngram_semantics=False,
        show_progress=True,
    )
    pipeline.induce()
    print_morphemes(pipeline=pipeline)
