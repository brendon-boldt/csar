from ..induction import MorfessorPipeline


def test_morfessor_pipeline() -> None:
    obs: list = [
        ((0, 1, 2), frozenset()),
        ((1, 2), frozenset()),
        ((1, 2, 3), frozenset()),
        ((5, 6, 7, 8, 9), frozenset()),
        ((0, 3), frozenset()),
    ]

    pipeline = MorfessorPipeline(obs)
    pipeline.induce()
    inventory = pipeline.get_morphemes()
    print(inventory)
    assert len(inventory) > 0
