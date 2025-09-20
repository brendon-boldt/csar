from typing import NamedTuple

import numpy as np

from ..common import sfrozenset

FormList = tuple[tuple[int, ...], ...]
MeaningSet = sfrozenset[sfrozenset[int]]
FMSetPair = tuple[FormList, MeaningSet]


class DatasetRecord(NamedTuple):
    form_tids: np.ndarray
    form_tids_original: np.ndarray
    meaning_tids: sfrozenset[int]
