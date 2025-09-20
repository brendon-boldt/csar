from typing import Iterable, Any, Callable, TypeVar
import os
import warnings
import time
import pprint

import numpy as np
import pytest
import joblib
import scipy

from .. import induction
from ..common import Observation, Morpheme
from .. import logutils
from .datasets import Case, get_mkdataset_params
from ..generation import dataset_list

logger = logutils.logger.getChild(__name__)

N_SEEDS = int(os.environ.get("N_SEEDS", 1))
random_seeds = pytest.mark.parametrize("seed", np.arange(N_SEEDS))


def do_morphology(
    case: Case,
    observations: Iterable[Observation],
    gt_morphemes: Iterable[Morpheme],
) -> Any:
    pipeline = induction.CsarPipeline(
        observations, max_ngram=999_999, max_semcomps=999_999
    )
    pipeline.induce()
    morphemes = pipeline.extractor.morphemes
    logger.debug("Ground Truth")
    logger.debug("\n" + pprint.pformat(gt_morphemes))
    logger.debug("Extracted")
    logger.debug("\n" + pprint.pformat(morphemes))
    morph_set = set(morphemes)
    gt_morph_set = set(gt_morphemes)
    return {
        "exact": morph_set == gt_morph_set,
        "superset": morph_set >= gt_morph_set,
        "subset": morph_set <= gt_morph_set,
    }


def get_ci(failures: int, total: int, confidence) -> tuple[float, float]:
    return scipy.stats.binomtest(failures, total).proportion_ci(confidence)


@pytest.mark.parametrize("case", get_mkdataset_params())
def test_procedural_dataset_ci(*, case: Case, pytestconfig: Any) -> None:
    confidence = pytestconfig.option.confidence
    max_test_time = pytestconfig.option.time_limit
    start_time = time.monotonic()

    total = 0
    failures = {k: 0 for k in ["exact", "superset", "subset"]}

    def keep_going() -> bool:
        if total == 0:
            return True
        assert (
            time.monotonic() - start_time < max_test_time
        ), f"Could not complete test in {max_test_time} seconds ({total} trials ran)."
        all_pass = True
        for k, v in case.tests.items():
            low, high = get_ci(failures[k], total, confidence)
            logger.debug(f"{low:.3f} -- {failures[k]/total:.3f} -- {high:.3f}")
            print(f"{low:.3f} < {failures[k]/total:.3f} < {high:.3f}")
            assert not case.fail_fast or failures[k] == 0
            assert (
                low <= v
            ), f"{k} check failed with {failures[k]} failures out of {total} trials."
            all_pass &= high < v
        return not all_pass

    seed_generator = np.random.default_rng(0)
    while keep_going():
        seed = int.from_bytes(seed_generator.bytes(8))
        pdataset = case.mkdataset(seed)
        observations = pdataset.get_observations()
        gt_morphemes = pdataset.get_morphemes()
        results = do_morphology(case, observations, gt_morphemes)
        total += 1
        for k, v in results.items():
            failures[k] += not v


T = TypeVar("T")


def do_in_new_proc(f: Callable[[], T]) -> T:
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="This process .* is multi-threaded",
    )
    return joblib.Parallel(n_jobs=2)(joblib.delayed(f)() for _ in [()])[0]


@random_seeds
@pytest.mark.parametrize("case", get_mkdataset_params(True))
def test_pdataset_determinism(*, case: Case, seed: int) -> None:
    def _get_vals() -> tuple[Any, Any]:
        pdataset = case.mkdataset(seed)
        obs_val = tuple(pdataset.get_observations())
        morph_val = tuple(pdataset.get_morphemes())
        return obs_val, morph_val

    orig_vals = _get_vals()
    new_vals = do_in_new_proc(_get_vals)
    assert new_vals[0] == orig_vals[0]
    assert new_vals[1] == orig_vals[1]


@random_seeds
@pytest.mark.parametrize("case", get_mkdataset_params(True))
def test_algo_determinism(*, case: Case, seed: int) -> None:
    def do_morph() -> induction.CsarPipeline:
        pdataset = case.mkdataset(seed)
        obss = pdataset.get_observations()
        pipeline = induction.CsarPipeline(obss, max_ngram=999_999, max_semcomps=999_999)
        pipeline.induce()
        return pipeline

    morphy1 = do_in_new_proc(do_morph)
    morphy2 = do_in_new_proc(do_morph)
    assert morphy1.extractor.morphemes == morphy2.extractor.morphemes


@random_seeds
@pytest.mark.parametrize("ds_args", dataset_list.get_dataset_args())
def test_proc_dataset_meanings(*, seed: int, ds_args: dataset_list.DatasetArgs) -> None:
    ds = ds_args.pdataset(seed=seed, **ds_args.pd_args)
    ds.get_observations()
    for f, m in ds.get_morphemes():
        assert len(m) > 0


# def test_mixed_compositional() -> None:
# def test_holistic() -> None:
