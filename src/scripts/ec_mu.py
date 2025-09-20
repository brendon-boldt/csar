import csv
from pathlib import Path
import json

from ..util import load_input_data, write_input_data
from ..induction import CsarPipeline
from .util import run

INPUT_DIR = Path("data/ec-shapeworld")


def _process_raw_data(path: Path) -> list[tuple[list, list]]:
    data: list = []
    with path.open() as fo:
        reader = csv.reader(fo)
        for row in reader:
            utt, meaning, _, _, split = row
            if split != "train":
                continue

            _utt = [int(x) for x in utt.split(" ")]
            _meaning = meaning.split(" ")
            data.append((_utt, _meaning))
    return data


def _get_variant(path: Path) -> str:
    if path.name == "sampled_lang.csv":
        return path.parents[0].name.split("_")[1]
    return path.stem


def get_data(path: Path) -> tuple[list, str]:
    try:
        return load_input_data(path), _get_variant(path)
    except json.JSONDecodeError:
        data = _process_raw_data(path)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        variant = _get_variant(path)
        new_path: Path = INPUT_DIR / f"{variant}.jsonl"
        write_input_data(data, new_path)
        return data, variant


def main(
    *,
    input_path: Path,
    max_lines: int | None,
    max_inventory_size: int | None,
    write_output: bool,
) -> None:
    observations, variant = get_data(input_path)
    # Remove beginning- and end-of-sentence tokens
    observations = [(f[1:-1], m) for f, m in observations]

    pipeline = CsarPipeline(
        observations[: max_lines or 20_000],
        max_ngram=999_999,
        trim_threshold=1,
        max_semcomps=999_999,
        max_inventory_size=max_inventory_size,
    )
    run(f"ec-shapeworld-{variant}", pipeline, write_output=write_output, do_toposim=True)
