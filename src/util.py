import json
from pathlib import Path
import io
import typing

import numpy as np
import scipy.stats
import Levenshtein

from .common import sfrozenset, Morpheme, Form, Meaning, InductionMorpheme
from .induction import CsarPipeline


def _check_input_data_format(data: typing.Any) -> None:
    if not isinstance(data, list):
        raise ValueError("Data is not list.")
    if not data:
        return
    if not isinstance(data[0], list):
        raise ValueError("List item is not a list.")
    if not len(data[0]) == 2:
        raise ValueError("List item is not a 2-tuple.")
    if not (isinstance(data[0][0], list) and isinstance(data[0][1], list)):
        raise ValueError("List item is not tuple of lists.")


def _load_input_data_json(file_object: io.TextIOWrapper) -> list:
    file_object.seek(0)
    data = json.load(file_object)
    _check_input_data_format(data)
    return data


def load_input_data(pathlike: str | Path) -> list:
    with open(pathlike) as fo:
        try:
            return _load_input_data_json(fo)
        except (json.JSONDecodeError, ValueError):
            fo.seek(0)
        line = fo.readline()
        datum = json.loads(line)
        _check_input_data_format([datum])
        data = [datum] + [json.loads(x) for x in fo]
    return data


def write_input_data(data: list, pathlike: str | Path) -> None:
    with open(pathlike, "w") as fo:
        fo.writelines(json.dumps([f, list(m)]) + "\n" for f, m in data)


def _form_sim(x: Form, y: Form) -> float:
    return Levenshtein.ratio(x, y)


def _meaning_sim(x: Meaning, y: Meaning) -> float:
    u = len(x | y)
    if u == 0:
        return 1
    return len(x & y) / u


# TODO Rename to fuzzy match or something.
def _best_match_morpheme(pair: Morpheme, inv: set[Morpheme]) -> float:
    best = 0.0
    form, meaning = pair
    for f2, m2 in inv:
        score = min(_form_sim(form, f2), _meaning_sim(meaning, m2))
        best = max(best, score)
        if best == 1.0:
            break
    return best


def inventory_similarity(
    inv1: typing.Iterable[Morpheme],
    inv2: typing.Iterable[Morpheme],
    *,
    exact: bool = False,
    form_only: bool = False,
) -> tuple[float, float, float]:

    meaning_constructor: typing.Any = (
        (lambda x: sfrozenset([0])) if form_only else sfrozenset
    )

    inv1 = set((tuple(f), meaning_constructor(m)) for f, m in inv1)
    inv2 = set((tuple(f), meaning_constructor(m)) for f, m in inv2)

    if not (inv1 and inv2):
        return 0, 0, 0

    def _in(x, y) -> float:
        return x in y

    match_func = _in if exact else _best_match_morpheme

    precision = sum(match_func(p, inv1) for p in inv2) / len(inv2)
    recall = sum(match_func(p, inv2) for p in inv1) / len(inv1)

    if recall == 0 or precision == 0:
        f_score = 0.0
    else:
        f_score = 2 / ((1 / precision) + (1 / recall))
    return precision, recall, f_score


def compute_toposim(_observations: typing.Iterable) -> float:
    observations = list(_observations)
    rng = np.random.default_rng()
    n_samples = 100000
    idxs = rng.choice(len(observations), (n_samples, 2))

    os = observations
    dists = [
        (
            _form_sim(os[i][0], os[j][0]),
            _meaning_sim(set(os[i][1]), set(os[j][1])),  # type: ignore[arg-type]
        )
        for i, j in idxs
    ]
    return scipy.stats.spearmanr(dists)[0]


def print_morphemes(
    induction_morphemes: typing.Iterable[InductionMorpheme] | None = None,
    *,
    pipeline: CsarPipeline | None = None,
) -> None:
    ims: typing.Iterable[InductionMorpheme]
    if pipeline is not None:
        ims = pipeline.extractor.induction_morphemes
    elif induction_morphemes is not None:
        ims = induction_morphemes
    else:
        raise ValueError()
    max_form_str_length = max(len(str(x)) for x in ims)
    for im in ims:
        print(
            f"{str(im):<{max_form_str_length}}  {im.prevalence:.3f} {-im.induced_weight:.3f} {-im.initial_weight:.3f}"
        )
