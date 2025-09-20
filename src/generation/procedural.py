from typing import Iterable, cast, Sequence, Iterator
import typing
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools

import numpy as np

from ..common import Observation, Token, Meaning, Form, Morpheme, sfrozenset


class ProceduralDataset(ABC):
    @abstractmethod
    def __init__(self, *, seed: int, **kwargs) -> None:
        self._observations: None | list[Observation] = None
        self.seed = seed

    @abstractmethod
    def _get_form_for_meaning(self, meaning: Meaning) -> Form:
        pass

    @abstractmethod
    def get_morphemes(self) -> Iterable[Morpheme]:
        pass

    @abstractmethod
    def _generate_meanings(self) -> Iterable[Meaning]:
        pass

    def get_observations(self) -> Iterable[Observation]:
        if self._observations is None:
            meanings = self._generate_meanings()
            self._observations = [(self._get_form_for_meaning(m), m) for m in meanings]
        return self._observations


to_tuple = np.frompyfunc(lambda x: (x,), 1, 1)


class Compositional(ProceduralDataset):
    def __init__(
        self,
        *,
        seed: int,
        samples: int = 300,
        attrs: int = 3,
        vals: int = 3,
        forms_per_meaning: int = 1,
        syn_balanced: bool = True,
        meaning_balanced: bool = True,
        bag_of_meanings: bool = False,
        polysemy_rate: float = 0,
        extra_form_rate: float = 0,
        extra_meaining_rate: float = 0,
        shuffle: bool = False,
        atoms_per_form: tuple[int, int] = (1, 1),
        vocab_size: None | int = None,
        apf_no_overlap: bool = False,
        mixed_comp: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        self.samples = samples
        self.vals = vals
        attrs = attrs if not bag_of_meanings else 1
        self.attrs = attrs
        self.forms_per_meaning = forms_per_meaning
        self.syn_balanced = syn_balanced
        self.meaning_balanced = meaning_balanced
        self.bag_of_meanings = bag_of_meanings
        self._morphemes: dict[sfrozenset[str], set[tuple[int, ...]]] = defaultdict(set)
        self.polysemy_rate = polysemy_rate
        self.extra_meaining_rate = extra_meaining_rate
        self.extra_form_rate = extra_form_rate
        self.mixed_comp = mixed_comp
        self.shuffle = shuffle
        # TODO Take this as an arg
        self.rng = np.random.default_rng(seed)

        self.apf_no_overlap = apf_no_overlap
        self.atoms_per_form = atoms_per_form
        assert all(x >= 1 for x in atoms_per_form)
        assert atoms_per_form[0] <= atoms_per_form[1]
        _mapping: np.ndarray
        if atoms_per_form == (1, 1):
            _mapping = np.arange(
                vals * attrs * forms_per_meaning, dtype=np.int64
            ).reshape(attrs, vals, forms_per_meaning)
            self.mapping = to_tuple(_mapping)
        else:
            n_forms = vals * attrs * forms_per_meaning
            if self.apf_no_overlap:
                vocab_size = n_forms * atoms_per_form[1]
            else:
                vocab_size = vocab_size or n_forms
            forms = self.rng.choice(
                vocab_size,
                (n_forms, atoms_per_form[1]),
                replace=not self.apf_no_overlap,
            )
            form_lens = self.rng.integers(
                atoms_per_form[0], atoms_per_form[1] + 1, n_forms
            )
            _mapping = np.full(n_forms, object(), dtype=object)
            for i in range(n_forms):
                _mapping[i] = tuple(forms[i][: form_lens[i]].tolist())
            self.mapping = _mapping.reshape(attrs, vals, forms_per_meaning)
        self.vocab_size = vocab_size or self.mapping.max()[0]
        self._apply_polysemy()

    def _apply_polysemy(self) -> None:
        mapping = self.mapping
        n_remappings = int(mapping.size * self.polysemy_rate)
        to_replace = self.rng.choice(mapping.size, n_remappings, replace=False)
        remains = np.delete(mapping.flatten(), to_replace)
        new_vals = self.rng.choice(remains, to_replace.size)
        mapping.reshape(-1)[to_replace] = new_vals

    def _get_form_for_meaning(self, _meaning: Meaning) -> Form:
        meaning = cast(Iterable[str], _meaning)
        form_comps: list[tuple[Token, ...]] = []
        fpm = self.forms_per_meaning
        for m in sorted(meaning):
            attr, val = [int(x) for x in m.split("_")]
            ramp_dist = (x := (1 + np.arange(fpm))) / x.sum()
            p = None if self.syn_balanced else ramp_dist
            syn_samp = self.rng.choice(fpm, p=p)
            form_comp = self.mapping[attr, val, syn_samp]
            form_comps.append(form_comp)
            self._morphemes[sfrozenset({m})].add(form_comp)
        # TODO Also, we might have start to accept "good enough" and determine
        # some kind of fitness of morphology.
        # TODO Add shuffle forms
        # logger.debug(f"{form} -> {meaning}")
        n_extra_forms = self.rng.geometric(1 - self.extra_form_rate) - 1
        if self.extra_form_rate == 0:
            assert n_extra_forms == 0
        extra_forms = to_tuple(
            self.rng.integers(0, self.vocab_size + 1, size=n_extra_forms)
        )
        for ef in extra_forms:
            idx = self.rng.integers(0, len(form_comps) + 1)
            form_comps.insert(idx, ef)
        if self.shuffle:
            form_comps = self.rng.permutation(np.fromiter(form_comps, dtype=object)).tolist()  # type: ignore
        form: typing.Any = tuple(x for t in form_comps for x in t)
        # logger.debug(f"{meaning} -> {form}")
        return form

    def get_morphemes(self) -> list[Morpheme]:
        return [(f, m) for m, fs in self._morphemes.items() for f in fs]

    def _generate_meanings(self) -> Iterator[Meaning]:
        ramp_dist = (x := (1 + np.arange(self.vals))) / x.sum()
        p = None if self.meaning_balanced else ramp_dist
        data: Iterable[Sequence]
        if self.bag_of_meanings:
            if p is None:
                p = np.array(0.5)
            samples = self.rng.random((self.samples, self.vals)) < p[None]
            data = [row.nonzero()[0] for row in samples]
            yield from (
                sfrozenset({f"0_{v}" for v in meaning})
                for meaning in data
                if len(meaning) > 0
            )
        else:
            data = self.rng.choice(self.vals, (self.samples, self.attrs), p=p)
            yield from (
                sfrozenset({f"{a}_{v}" for a, v in enumerate(meaning)})
                for meaning in data
            )


class MixedComp(Compositional):
    def __init__(
        self,
        *,
        seed: int,
        samples: int = 300,
        attrs: int = 3,
        vals: int = 3,
        bag_of_meanings: bool = False,
        atoms_per_form: tuple[int, int] = (1, 1),
        apf_no_overlap: bool = False,
        vocab_size: None | int = None,
        meaning_balanced: bool = True,
        shuffle: bool = False,
        extra_form_rate: float = 0,
    ) -> None:
        ProceduralDataset.__init__(self, seed=seed)
        self.samples = samples
        self.vals = vals
        attrs = attrs if not bag_of_meanings else 1
        self.attrs = attrs
        self.meaning_balanced = meaning_balanced
        self.bag_of_meanings = bag_of_meanings
        self._morphemes: dict[sfrozenset[str], set[tuple[int, ...]]] = defaultdict(set)
        self.shuffle = shuffle
        self.extra_form_rate = extra_form_rate
        self.rng = np.random.default_rng(seed)

        x, y = (1, self.vals) if self.bag_of_meanings else (self.vals, self.attrs)
        _idxs = np.tile(np.arange(-1, x), (y, 1))
        _prod = [
            tuple(
                (0, i) if self.bag_of_meanings else (i, int(x))
                for i, x in enumerate(y)
                if x != -1
            )
            for y in itertools.product(*_idxs)
        ]
        self.rng.shuffle(_prod)
        meaning_groups = _prod

        self.apf_no_overlap = apf_no_overlap
        self.atoms_per_form = atoms_per_form
        assert all(x >= 1 for x in atoms_per_form)
        assert atoms_per_form[0] <= atoms_per_form[1]
        n_forms = len(meaning_groups)
        if self.apf_no_overlap:
            vocab_size = n_forms * atoms_per_form[1]
        else:
            vocab_size = vocab_size or n_forms
        forms = self.rng.choice(
            vocab_size,
            (n_forms, atoms_per_form[1]),
            replace=not self.apf_no_overlap,
        )
        form_lens = self.rng.integers(atoms_per_form[0], atoms_per_form[1] + 1, n_forms)
        _mapping = np.full(n_forms, object(), dtype=object)
        for i in range(n_forms):
            _mapping[i] = tuple(forms[i][: form_lens[i]].tolist())
        self.mapping = {m: f for m, f in zip(meaning_groups, _mapping)}
        self.vocab_size = vocab_size

    def _get_form_for_meaning(self, _meaning: Meaning) -> Form:
        form_comps: list[tuple[Token, ...]] = []
        meaning = np.zeros((self.attrs, self.vals), dtype=bool)
        meaning[
            tuple(
                zip(*[tuple(int(y) for y in cast(str, x).split("_")) for x in _meaning])
            )
        ] = True

        for mg, form_comp in self.mapping.items():
            if not meaning.any():
                break
            mg_idxr = tuple(zip(*mg))
            if not meaning[mg_idxr].all():
                continue
            meaning[mg_idxr] = False
            form_comps.append(form_comp)
            m = [f"{a}_{v}" for a, v in mg]
            self._morphemes[sfrozenset(m)].add(form_comp)
        assert not meaning.any()
        n_extra_forms = self.rng.geometric(1 - self.extra_form_rate) - 1
        if self.extra_form_rate == 0:
            assert n_extra_forms == 0
        extra_forms = to_tuple(
            self.rng.integers(0, self.vocab_size + 1, size=n_extra_forms)
        )
        for ef in extra_forms:
            idx = self.rng.integers(0, len(form_comps) + 1)
            form_comps.insert(idx, ef)
        if self.shuffle:
            form = self.rng.permutation(np.fromiter(form_comps, dtype=object)).tolist()
        form = tuple(x for t in form_comps for x in t)
        return form
