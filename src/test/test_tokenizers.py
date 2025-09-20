import pprint

from .. import common
from ..induction import TokenizerPipeline


def test_bpe() -> None:
    obs: list = [
        ((0, 1, 2), frozenset()),
        ((1, 2), frozenset()),
        ((1, 2, 3), frozenset()),
        ((5, 6, 7, 8, 9), frozenset()),
        ((5, 6, 7, 8, 9, 0), frozenset()),
        ((0, 3), frozenset()),
    ]

    sfs = common.sfrozenset
    gt_morphemes: list = [  # noqa: F841
        ((0,), sfs()),
        ((1, 2), sfs()),
        ((3,), sfs()),
        ((5, 6, 7, 8, 9), sfs()),
    ]
    # pipeline = TokenizerPipeline(obs, gt_morphemes=gt_morphemes, model="unigram")
    # pipeline = TokenizerPipeline(obs, gt_morphemes=gt_morphemes, model="bpe")
    pipeline = TokenizerPipeline(obs, vocab_size=8, model="unigram")
    pipeline.induce()

    pprint.pprint(pipeline.get_morphemes())

    # toker = tokenizers.CharBPETokenizer(
    #     dropout=0,
    #     bert_normalizer=False,
    # )

    # print(_corpus)
    # toker.train_from_iterator(
    #     # _corpus,
    #     # ["our father who are in heaven hallowed be thy name thy kingdome come thy will be done on earth as it is in heaven"],
    #     _corpus,
    #     vocab_size=100,
    #     suffix="",
    #     show_progress=False,
    #     limit_alphabet=999_999_999,
    #     min_frequency=0,
    # )
    # print(toker.get_vocab())
    # rvocab = {v: k for k, v in toker.get_vocab().items()}
    # from pprint import pprint
    # pprint(sorted(rvocab.items(), key=lambda x: x[0]))
    # breakpoint()

    # print(toker.encode("AB").tokens)
    # print(toker.encode("AB").ids)
