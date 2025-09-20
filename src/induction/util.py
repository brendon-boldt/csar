from typing import Iterable, Sequence
import itertools

from .common import FormList
from ..common import sfrozenset


def get_ngrams(tokens_ids: Sequence[int], max_size: int) -> FormList:
    unigrams = [(x,) for x in tokens_ids]
    ngrams = [
        tuple(tokens_ids[i : i + n])
        for n in range(2, min(max_size, len(tokens_ids)) + 1)
        for i in range(len(tokens_ids) - n + 1)
    ]
    # return frozenset({()}) | unigrams | ngrams
    return tuple(x for x in unigrams + ngrams if -1 not in x)


def powerset(vals: Iterable[int], max_size: int) -> sfrozenset[sfrozenset[int]]:
    vals = list(vals)
    return sfrozenset(
        sfrozenset(c)
        for r in range(1, min(max_size, len(vals)) + 1)
        for c in itertools.combinations(vals, r)
    )
