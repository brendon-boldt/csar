import typing

import tokenizers

from .pipeline import Pipeline
from .. import common

ModelType = typing.Literal["bpe", "unigram"]


OFFSET = 256


def ints2chars(ints: typing.Iterable) -> str:
    return "".join(chr(x + OFFSET) for x in ints)


def chars2ints(chars: str) -> tuple[int, ...]:
    return tuple(ord(x) - OFFSET for x in chars)


class TokenizerPipeline(Pipeline):
    def __init__(
        self,
        observations: typing.Iterable[common.Observation],
        model: ModelType,
        *,
        vocab_size: int | None = None,
        gt_morphemes: typing.Iterable[common.Morpheme] | None = None,
    ) -> None:
        super().__init__(observations)
        if vocab_size is None and gt_morphemes is None:
            raise ValueError("Must specify vocab_size or gt_morphemes")

        self.vocab_size = vocab_size
        self.gt_morphemes = gt_morphemes
        self.model = model
        self.tokenizer_class = (
            tokenizers.models.BPE if self.model == "bpe" else tokenizers.models.Unigram
        )

    def _make_tokenizer(
        self, raw_corpus: typing.Iterable, vocab_size: int
    ) -> tokenizers.Tokenizer:
        corpus = [ints2chars(utt) for utt, _ in raw_corpus]
        toker = tokenizers.Tokenizer(self.tokenizer_class())
        match self.model:
            case "bpe":
                trainer_class = tokenizers.trainers.BpeTrainer
            case "unigram":
                trainer_class = tokenizers.trainers.UnigramTrainer
        trainer = trainer_class(
            vocab_size=vocab_size,
            show_progress=False,
            **(
                {"min_frequency": 0, "limit_alphabet": 999_999_999}
                if self.model == "bpe"
                else {}
            ),
        )
        toker.train_from_iterator(corpus, trainer=trainer)
        return toker

    def induce_vocab_size(self) -> int:
        assert self.gt_morphemes is not None
        toker = self._make_tokenizer(self.gt_morphemes, 999_999_999)
        return len(toker.get_vocab())

    def induce(self) -> list[common.Morpheme]:
        vocab_size = self.vocab_size or self.induce_vocab_size()
        if self.model == "unigram":
            vocab_size = max(
                vocab_size, len({y for x, _ in self.observations for y in x}) + 1
            )

        toker = self._make_tokenizer(self.observations, vocab_size)

        self.morphemes: list[tuple[tuple[int, ...], common.sfrozenset]] = list(
            {
                (chars2ints(token), common.sfrozenset())
                for utt, _ in self.observations
                if utt != "<unk>"
                for token in toker.encode(ints2chars(utt)).tokens
            }
        )

        return self.get_morphemes()

    def get_morphemes(self) -> list[common.Morpheme]:
        return self.morphemes  # type: ignore
