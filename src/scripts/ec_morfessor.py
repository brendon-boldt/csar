from pathlib import Path
import json

import pandas as pd

from ..induction import MorfessorPipeline
from . import ec_mu



def run(input_path: Path, max_lines: int) -> dict:
    print(f"Running Morfessor on {input_path}.")
    which = input_path.parents[0].name
    match which:
        case "ec-vector":
            data: list = json.loads(input_path.read_text())
            dataset = data[:max_lines]
            # Remove end-of-sentence padding
            dataset = [([y for y in f if y != 0], m) for f, m in dataset]
            variant = input_path.stem
        case "ec-shapeworld":
            dataset, variant = ec_mu.get_data(input_path)
            dataset = dataset[:max_lines]
            # Remove beginning- and end-of-sentence tokens
            dataset = [(f[1:-1], m) for f, m in dataset]
        case _:
            raise ValueError(which)

    pipeline = MorfessorPipeline(dataset)
    pipeline.induce()
    morphemes = pipeline.get_morphemes()

    return {
        "name": variant,
        "inv_size": len(morphemes),
        "mean_length": sum(len(x[0]) for x in morphemes) / len(morphemes),
    }


def main(
    *,
    max_lines: int | None = None,
) -> None:
    input_paths = (
        "ec-vector/av.json",
        "ec-vector/sparse.json",
        "ec-shapeworld/ref.jsonl",
        "ec-shapeworld/setref.jsonl",
        "ec-shapeworld/concept.jsonl",
    )

    records = [run(Path("data") / p, max_lines or 20_000) for p in input_paths]
    df = pd.DataFrame.from_records(records)
    print(df)
    df.to_latex(Path("output") / "ec-morfessor-table.tex", float_format="%.2f")
