from pathlib import Path
from typing import Any, Sequence, Iterator
import json
from pprint import pprint

import pytest

from ..induction import CsarPipeline
from ..common import Morpheme, Observation, sfrozenset


def get_case_paths() -> Iterator[Any]:
    case_path_root = Path("src/test/data/morphology")
    for p in case_path_root.glob("**/*.json"):
        marks = []
        if p.parents[0].stem == "xfail":
            marks.append(pytest.mark.xfail())
        yield pytest.param(p, marks=marks)


def preprocess_gt_morphemes(raw_morphs: Any) -> Sequence[Morpheme]:
    return [(tuple(f.split(" ")), sfrozenset(m)) for f, m in raw_morphs]


def preprocess_observations(
    observations: list[tuple[str, list]],
) -> Sequence[Observation]:
    return [(tuple(utt.split(" ")), sfrozenset(m)) for utt, m in observations]


def parametrize_test_files() -> Any:
    return pytest.mark.parametrize(
        "data_path", cps := list(get_case_paths()), ids=[x.values[0].stem for x in cps]
    )


@parametrize_test_files()
def test_handcrafted(data_path: Path) -> None:
    with data_path.open() as fo:
        data = json.load(fo)

    observations = preprocess_observations(data["observations"])
    gt_morphemes = preprocess_gt_morphemes(data["morphemes"])

    pipeline = CsarPipeline(observations, max_ngram=9999, max_semcomps=9999)
    pipeline.induce()
    assert (morphemes := pipeline.extractor.morphemes)

    pprint(morphemes)
    assert set(morphemes) == set(gt_morphemes)
