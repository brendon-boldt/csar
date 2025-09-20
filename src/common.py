from typing import (
    TypedDict,
    Any,
    TypeVar,
    Generic,
    Iterator,
    TYPE_CHECKING,
)
import argparse
import time
import inspect
import dataclasses
import logging

if TYPE_CHECKING:
    from _typeshed import SupportsDunderLT

SortableT = TypeVar("SortableT", bound="SupportsDunderLT", covariant=True)


class sfrozenset(Generic[SortableT], frozenset[SortableT]):
    def __iter__(self) -> Iterator[SortableT]:
        return iter(sorted(super().__iter__()))


x: sfrozenset[int] = sfrozenset([])
y: sfrozenset[int | str] = x

a: frozenset[int] = frozenset([])
b: frozenset[int | str] = a


class AnnotatedUtterance(TypedDict):
    utterance: str
    semantics: list[str]


class ImageData(TypedDict):
    captions: list[str]
    instances: list[str]


ImageId = int
ImageMap = dict[ImageId, ImageData]
Dataset = list[AnnotatedUtterance]
Token = str | int | bool

# Form--meaning pair
Observation = tuple[tuple[Token, ...], sfrozenset[Token]]
Form = tuple[Token, ...]
Meaning = sfrozenset[Token]
Morpheme = tuple[Form, Meaning]


@dataclasses.dataclass()
class InductionMorpheme:
    form: Form
    meaning: Meaning
    initial_weight: float
    induced_weight: float
    prevalence: float

    def to_morpheme(self) -> Morpheme:
        return (self.form, self.meaning)

    def __str__(self) -> str:
        form = " ".join(str(x) for x in self.form)
        meaning = ", ".join(str(x) for x in self.meaning)
        return f"({form}, {{{meaning}}})"

    def _sort_key(self) -> tuple:
        return self.initial_weight, self.induced_weight, self.prevalence

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, InductionMorpheme):
            raise ValueError()
        return self._sort_key() < other._sort_key()


T = TypeVar("T")


@dataclasses.dataclass(order=True)
class PrioritizedItem(Generic[T]):
    priority: float | tuple[float, ...]
    value: T = dataclasses.field(compare=False)


def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", type=str)
    ap.add_argument("-i", "--input-path", type=str, default=None)
    ap.add_argument("-t", "--max-tuple", type=int, default=1)
    ap.add_argument("-n", "--max-ngram", type=int, default=1)
    ap.add_argument("-o", "--output-path", type=str, default=None)
    return ap.parse_args()


_last_ts = time.perf_counter()


def time_here(s: Any = "") -> None:
    global _last_ts
    now = time.perf_counter()
    elapsed = now - _last_ts
    print(f"At {str(s)}: {elapsed:.1f}s")
    _last_ts = now


def logvar(name: str) -> None:
    locals = inspect.currentframe().f_back.f_locals  # type: ignore
    varstr = str(locals[name])
    multiline = "\n" if "\n" in varstr else " "
    logging.debug(f"{name} ={multiline}{varstr}")  # type: ignore
