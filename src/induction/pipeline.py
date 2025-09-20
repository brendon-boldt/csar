import typing
import abc

from .. import common


class Pipeline(abc.ABC):
    def __init__(
        self,
        observations: typing.Iterable[common.Observation],
    ) -> None:
        self.observations = observations

    @abc.abstractmethod
    def induce(self) -> list[common.Morpheme]: ...

    @abc.abstractmethod
    def get_morphemes(self) -> list[common.Morpheme]: ...
