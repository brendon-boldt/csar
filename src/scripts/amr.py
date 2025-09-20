import re
from pathlib import Path

import nltk

from .. import induction
from .. import common
from .util import run


def parse_amr_file(
    _path: str | Path,
) -> list[tuple[tuple[str], common.sfrozenset[str]]]:
    path = Path(_path)
    tok = nltk.tokenize.RegexpTokenizer(r"\w+").tokenize

    snt_pattern = r"^# ::snt (.*)$"
    arg_pattern = r"/ ([-a-z0-9]+)"

    with open(path) as fo:
        pairs: list = []
        while True:
            while True:
                line = fo.readline()
                if not line:
                    return pairs
                if m := re.match(snt_pattern, fo.readline()):
                    snt = tok(m[1])
                    break
            args = set()
            while (line := fo.readline()).strip():
                if line[0] == "#":
                    continue
                args.update(re.findall(arg_pattern, line))
            pairs.append((tuple(snt), common.sfrozenset(args)))
    return pairs


def main(
    *,
    input_path: Path,
    max_lines: int | None,
    max_inventory_size: int | None,
    write_output: bool,
) -> None:
    observations = parse_amr_file(input_path)[:max_lines]

    pipeline = induction.CsarPipeline(
        observations,
        max_ngram=3,
        max_semcomps=3,
        trim_threshold=4,
        search_best_sub=False,
        max_inventory_size=max_inventory_size,
        vocab_size=10_000,
        token_vocab_size=500,
    )
    run("amr", pipeline, write_output=write_output)
