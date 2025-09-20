from typing import Any

import pytest
from pathlib import Path

from .. import util
from ..scripts.amr import parse_amr_file


EPSILON = 1e-6


def test_best_match_morpheme() -> None:
    inv1: tuple[Any, ...] = (
        ((0,), {0}),
        ((0, 1), {0, 1}),
        (
            (
                2,
                3,
                4,
            ),
            {2, 3, 4},
        ),
    )
    cases: tuple[Any, ...] = (
        (((), set()), (), 0),
        (((), set()), inv1, 0),
        ((("x",), {"x"}), inv1, 0),
        (([0], {0}), inv1, 1),
        (([0, 1], {0, 1}), inv1, 1),
        (([0], {2, 3, 4}), inv1, 0),
    )

    for pair, inv, result in cases:
        assert abs(util._best_match_morpheme(pair, inv) - result) < EPSILON


def test_inventory_similarity() -> None:
    inv1: tuple[Any, ...] = (
        ((0,), {0}),
        ((0, 1), {0, 1}),
        (
            (
                2,
                3,
                4,
            ),
            {2, 3, 4},
        ),
    )
    inv2: tuple[Any, ...] = (((0,), {0}),)
    inv2
    added: tuple[Any, ...] = inv1 + (((9,), {9}),)
    _deleted = list(inv1)
    del _deleted[0]
    deleted = tuple(_deleted)

    cases: tuple[tuple[Any, Any, float], ...] = (
        ((), (), 0),
        (inv1, (), 0),
        ((), inv1, 0),
        (inv1, inv1, 1),
        (inv1, added, 0.8571428571428572),
        (inv1, deleted, 0.9411764705882353),
        (inv1, inv2, 0.7142857142857143),
    )

    for x, y, val in cases:
        out_val = util.inventory_similarity(x, y)[2]
        assert abs(out_val - val) < EPSILON


def test_parse_amr() -> None:
    path = "data/amr/amr-release-3.0-amrs-dfa.txt"
    parse_amr_file(path)


def test_load_input_data(tmp_path: Path) -> None:
    file_path = tmp_path / "test_load_input_data"

    bad_data = [
        "",
        "[",
        "[3]",
        "{}",
        "[[], {}]",
        "[[]]",
        "[[]]\n[",
        "[]\n[]",
    ]

    good_data = [
        "[]",
        "[[],[]]",
        "[[[],[]]]",
        "[[[],[]],[[],[]]]",
        "[[],[]]\n[[],[]]",
    ]

    for datum in bad_data:
        with file_path.open("w") as fo:
            fo.write(datum)
        with pytest.raises(ValueError):
            util.load_input_data(file_path)
    for datum in good_data:
        with file_path.open("w") as fo:
            fo.write(datum)
        print(datum)
        res = util.load_input_data(file_path)
        assert isinstance(res, list)
