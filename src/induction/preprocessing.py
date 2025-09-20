from pathlib import Path
from typing import TypeVar
import argparse
import itertools
import json
import pickle as pkl

import scipy.sparse
import numpy as np

from ..common import Dataset


T = TypeVar("T")


def load_dataset(path: Path | str) -> Dataset:
    with open(path) as fo:
        return json.load(fo)


def powerset(vals: list[T], max_size: int) -> list[tuple[T, ...]]:
    return [c for r in range(1, max_size + 1) for c in itertools.combinations(vals, r)]


def count_coocs(args: argparse.Namespace, dataset: Dataset) -> None:
    form_set: set[str] = set()
    meaning_set: set[tuple[str, ...]] = set()

    def get_ngrams(utt: str) -> list[str]:
        unigrams = utt.split(" ")
        ngrams = [
            " ".join(unigrams[i : i + n])
            for n in range(2, args.max_ngram + 1)
            for i in range(len(unigrams) - n + 1)
        ]
        return unigrams + ngrams

    Pair = tuple[list[str], list[tuple[str, ...]]]
    pairs: list[Pair] = []

    for au in dataset:
        ngrams = get_ngrams(au["utterance"])
        form_set.update(ngrams)
        semantics = powerset(au["semantics"], args.max_tuple)
        meaning_set.update(semantics)
        pairs.append((ngrams, semantics))

    form_indexer = {v: i for i, v in enumerate(form_set)}
    form_rindexer = {i: v for i, v in enumerate(form_set)}
    meaning_indexer = {v: i for i, v in enumerate(meaning_set)}
    meaning_rindexer = {i: v for i, v in enumerate(meaning_set)}

    n_forms = len(form_indexer)
    n_meanings = len(meaning_indexer)

    type sparray = scipy.sparse.sparray

    def count_(_pairs: list[Pair]) -> "tuple[sparray, sparray, sparray]":
        n_pairs = len(_pairs)
        dok = scipy.sparse.dok_array
        form_occs = dok(
            (
                n_pairs,
                n_forms,
            ),
            dtype=bool,
        )
        meaning_occs = dok(
            (
                n_pairs,
                n_meanings,
            ),
            dtype=bool,
        )
        for i, p in sorted(list(enumerate(_pairs)), key=lambda x: x[1]):
            for x in p[0]:
                j = form_indexer[x]
                form_occs[i, j] = True
            for y in p[1]:
                k = meaning_indexer[y]
                meaning_occs[i, k] = True

        cooc = scipy.sparse.dok_array((n_forms, n_meanings), dtype=int)
        form_occs = form_occs.tocoo()
        meaning_occs = meaning_occs.tocoo()
        cooc = form_occs.T.astype(int) @ meaning_occs
        return (
            cooc,
            form_occs,
            meaning_occs,
        )

    coocs, form_occs, meaning_occs = count_(pairs)

    summed_form_occs = form_occs.sum(0)
    summed_meaning_occs = meaning_occs.sum(0)

    joint_ps = coocs.astype(float) / len(dataset)
    form_ps = summed_form_occs.astype(float) / len(dataset)
    meaning_ps = summed_meaning_occs.astype(float) / len(dataset)

    with open(args.output_path, "wb") as fo:
        pkl.dump(
            {
                "form_occs": form_occs.astype(np.int32),
                "meaning_occs": meaning_occs.astype(np.int32),
                "joint_ps": joint_ps.astype(np.float32),
                "form_ps": form_ps.astype(np.float32),
                "meaning_ps": meaning_ps.astype(np.float32),
                "dataset_length": len(dataset),
                "form_indexer": form_indexer,
                "form_rindexer": form_rindexer,
                "meaning_indexer": meaning_indexer,
                "meaning_rindexer": meaning_rindexer,
            },
            fo,
        )
