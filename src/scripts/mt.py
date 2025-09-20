import datasets
import nltk

from ..induction import CsarPipeline
from .util import run


def main(
    max_lines: None | int, max_inventory_size: None | int, write_output: bool
) -> None:
    dataset = datasets.load_dataset("wmt/wmt16", "de-en")

    tok = nltk.tokenize.RegexpTokenizer(r"\w+").tokenize

    try:
        tok("")
    except LookupError:
        nltk.download("punkt_tab")

    max_lines = max_lines or 20_000
    observations = [
        (tok(row["en"]), tok(row["de"]))
        for row in dataset["train"][:max_lines]["translation"]
    ]

    pipeline = CsarPipeline(
        observations,
        max_ngram=3,
        max_semcomps=3,
        trim_threshold=100,
        search_best_sub=False,
        max_inventory_size=max_inventory_size or 300,
        vocab_size=100_000,
        token_vocab_size=500,
        ngram_semantics=True,
    )
    run("mt", pipeline, write_output=write_output)
