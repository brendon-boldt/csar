import pprint

from ..induction import IbmPipeline


def test_ibm_model() -> None:
    # TODO Use CI-based testing
    # pdataset = case.mkdataset(seed)
    # observations = pdataset.get_observations()
    # gt_morphemes = pdataset.get_morphemes()

    o1: list = [
        ((0,), frozenset([1, 2])),
        ((0, 1), frozenset([1, 2, 10])),
        ((1, 2, 3), frozenset([10, 23])),
        ((2, 3), frozenset([23])),
    ]

    observations: list = [
        ((0, 7), frozenset([0])),
        ((7, 0, 1), frozenset([0, 1])),
        ((7, 1, 2, 3), frozenset([1, 23])),
        ((2, 3), frozenset([23])),
    ]

    for obs in [o1, observations]:
        morphemes = IbmPipeline(obs, model="model1").induce()
        pprint.pprint(morphemes)
        print()
