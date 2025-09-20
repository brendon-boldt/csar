import typing
import nltk

from .pipeline import Pipeline
from .. import common


ModelType = typing.Literal["model1", "model3"]


class IbmPipeline(Pipeline):
    def __init__(
        self, observations: typing.Iterable[common.Observation], *, model: ModelType
    ) -> None:
        super().__init__(observations)
        self.model = model

    def induce(self) -> list[common.Morpheme]:
        bitext = [
            nltk.translate.api.AlignedSent(list(f), list(m))
            for f, m in self.observations
        ]
        match self.model:
            case "model1":
                nltk.translate.ibm1.IBMModel1(bitext, 1)
            case "model3":
                nltk.translate.ibm3.IBMModel3(bitext, 1)

        morphemes: set = set()
        for sent in bitext:
            last_mi = -1
            current_f = []
            sorted_alignment = sorted(sent.alignment, key=lambda x: x[0])
            for fi, mi in sorted_alignment:
                form = sent.words[fi]
                if last_mi == -1 or mi == last_mi:
                    current_f.append(form)
                else:
                    if last_mi is not None:
                        meaning = common.sfrozenset([sent.mots[last_mi]])
                        morphemes.add((tuple(current_f), meaning))
                    current_f = [form]
                last_mi = mi
            if last_mi is not None:
                meaning = common.sfrozenset([sent.mots[last_mi]])
                morphemes.add((tuple(current_f), meaning))
        self.morphemes = list(morphemes)

        return self.get_morphemes()

    def get_morphemes(self) -> list[common.Morpheme]:
        return self.morphemes
