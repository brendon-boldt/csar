from pathlib import Path
import json
import pprint

from ..induction import CsarPipeline
from .util import run


def visualize(input_path: Path, *, max_lines: int | None) -> None:
    with input_path.open() as fo:
        data: list = json.load(fo)
    if max_lines:
        data = data[:max_lines]
    pprint.pprint(data)


def main(
    input_path: Path,
    *,
    max_lines: int | None,
    write_output: bool,
    max_inventory_size: int | None,
) -> None:
    with input_path.open() as fo:
        data: list = json.load(fo)
    dataset = data[:max_lines]
    # Remove end-of-sentence padding
    dataset = [([y for y in f if y != 0], m) for f, m in dataset]

    pipeline = CsarPipeline(
        dataset, max_ngram=99, max_semcomps=99, max_inventory_size=max_inventory_size
    )
    variant = input_path.stem
    run(f"ec-vector-{variant}", pipeline, write_output=write_output, do_toposim=True)
