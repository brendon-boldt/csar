from pathlib import Path
import typing
import functools
import itertools
import random
import json

import numpy as np
from tqdm import tqdm
import joblib
import pandas as pd

from .. import common
from ..generation.dataset_list import get_dataset_args, DatasetArgs

from .. import util
from ..induction.csar import logger as morphology_logger
from ..logutils import logging
from ..induction import CsarPipeline, MorfessorPipeline, TokenizerPipeline, IbmPipeline
from ..generation import ProceduralDataset

DUMP_DIR = Path("./output/procedural-dump")
PATH_F_SCORES = Path("save/analysis/f_scores.pkl")


class JSONEncoder(json.JSONEncoder):
    def default(self, obj: object) -> typing.Any:
        if isinstance(obj, set) or isinstance(obj, common.sfrozenset):
            return list(obj)
        return super().default(obj)


def _dump_result(name: str, pdataset: ProceduralDataset, extracted_morphemes) -> None:
    name = f"{name}-" + f"{pdataset.seed:04x}"[-4:]

    sim = functools.partial(
        util.inventory_similarity, pdataset.get_morphemes(), extracted_morphemes
    )
    metrics_fuzzy = sim(exact=False)
    metrics_exact = sim(exact=True)

    data = {
        "metrics": {
            "fuzzy": {
                "precision": metrics_fuzzy[0],
                "recall": metrics_fuzzy[1],
                "f1": metrics_fuzzy[2],
            },
            "exact": {
                "precision": metrics_exact[0],
                "recall": metrics_exact[1],
                "f1": metrics_exact[2],
            },
        },
        "observations": pdataset.get_observations(),
        "morphemes": {
            "ground_truth": pdataset.get_morphemes(),
            "extracted": extracted_morphemes,
        },
    }

    with (DUMP_DIR / f"{name}.json").open("w") as fo:
        json.dump(data, fo, cls=JSONEncoder)


def _compute_metric(ds_args: DatasetArgs, seed: int) -> list[dict]:
    morphology_logger.setLevel(logging.WARNING)

    pdataset = ds_args.pdataset(seed=seed, **ds_args.pd_args)
    observations = pdataset.get_observations()
    gt_morphemes = pdataset.get_morphemes()

    csar = CsarPipeline(observations, max_ngram=999_999, max_semcomps=999_999).induce()

    csar_results = util.inventory_similarity(gt_morphemes, csar, exact=True)
    if csar_results[-1] < 1:
        _dump_result(ds_args.name, pdataset, csar)

    morfessor = MorfessorPipeline(observations).induce()
    ibm1 = IbmPipeline(observations, model="model1").induce()
    ibm3 = IbmPipeline(observations, model="model3").induce()

    n_unique_tokens = len({f for fs, _ in observations for f in fs})
    vocab_size = (
        int(
            np.mean([len(f) / len(m) for f, m in observations])
            * len({m for _, ms in observations for m in ms})
        )
        + n_unique_tokens
    )

    bpe_fixed = TokenizerPipeline(
        observations, vocab_size=vocab_size, model="bpe"
    ).induce()
    unigram_fixed = TokenizerPipeline(
        observations, vocab_size=vocab_size, model="unigram"
    ).induce()

    out_obs = set(observations)

    funcs: list[tuple] = [
        ("fuzzy", util.inventory_similarity),
        ("exact", functools.partial(util.inventory_similarity, exact=True)),
        ("fuzzy_form", functools.partial(util.inventory_similarity, form_only=True)),
        (
            "exact_form",
            functools.partial(util.inventory_similarity, exact=True, form_only=True),
        ),
    ]
    results = []

    outputs = [
        ("csar", csar),
        ("obs", out_obs),
        ("morfessor", morfessor),
        ("bpe_heuristic", bpe_fixed),
        ("unigram_heuristic", unigram_fixed),
        ("ibm1", ibm1),
        ("ibm3", ibm3),
    ]

    for model, output in outputs:
        for name, f in funcs:
            vals = f(gt_morphemes, output)
            for name2, val in zip(["precision", "recall", "f1"], vals):
                results.append(
                    dict(
                        value=val,
                        model=model,
                        metric=f"{name}_{name2}",
                        dataset=ds_args.name,
                        seed=seed,
                    )
                    | ds_args.pd_args
                )

    return results


def collect(*, n_samples: int, overwrite: bool, n_jobs: int) -> None:
    if PATH_F_SCORES.exists() and not overwrite:
        return

    for p in DUMP_DIR.glob("*"):
        p.unlink()
    DUMP_DIR.mkdir(parents=True, exist_ok=True)

    argss = list(get_dataset_args())
    rng = np.random.default_rng(0)
    seeds = rng.choice(np.iinfo(np.int64).max, (len(argss), n_samples))

    jobs = [
        joblib.delayed(_compute_metric)(case, seeds[i, j])
        for i, case in enumerate(argss)
        for j in range(n_samples)
    ]
    random.shuffle(jobs)
    _results = joblib.Parallel(n_jobs=n_jobs)(tqdm(jobs))
    results = itertools.chain(*_results)
    df = pd.DataFrame.from_records(results)

    PATH_F_SCORES.parents[0].mkdir(exist_ok=True, parents=True)
    df.to_pickle(PATH_F_SCORES)
