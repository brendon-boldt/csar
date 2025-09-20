from pathlib import Path
import csv

from ..induction import CsarPipeline
from .util import run


def read_file(path: Path) -> list:
    dataset = []
    with path.open() as fo:
        reader = csv.reader(fo)
        for line in reader:
            form = ["^"] + list(line[0]) + ["$"]
            meaning = line[1].split(" ")
            dataset.append((form, meaning))
    return dataset


def main(
    max_lines: int | None, max_inventory_size: int | None, write_output: bool
) -> None:
    data_path = Path("./data/morpho-challenge.csv")
    dataset = read_file(data_path)
    dataset = dataset[:max_lines]

    pipeline = CsarPipeline(
        dataset,
        max_ngram=9999,
        max_semcomps=9999,
        max_inventory_size=max_inventory_size,
    )
    run("morpho-challenge", pipeline, write_output=write_output, no_form_space=True)
