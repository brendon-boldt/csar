from typing import Any, Iterator, TypeVar, Generic, Literal
import os

import numpy as np
import pytest

from ..generation.procedural import ProceduralDataset, Compositional, MixedComp

N_SEEDS = int(os.environ.get("N_SEEDS", 1))
random_seeds = pytest.mark.parametrize("seed", np.arange(N_SEEDS))

PD = TypeVar("PD", bound=ProceduralDataset)

CheckType = Literal["exact", "subset", "superset"]


class Case(Generic[PD]):
    def __init__(
        self,
        name: str,
        pdataset: type[PD],
        pd_args: dict,
        *,
        xfail: bool = False,
        tests: dict[CheckType, float] | None = None,
    ) -> None:
        # TODO The default should be fail if even one trial fails.
        self.fail_fast = not tests
        tests = tests or {}
        self.tests = {"exact": 0.05, **tests}
        self.xfail = xfail
        self.name = name
        self.pdataset = pdataset
        self.pd_args = pd_args

    def mkdataset(self, seed: int) -> PD:
        return self.pdataset(seed=seed, **self.pd_args)


def mkparam(case: Case) -> Any:
    marks = []
    if case.xfail:
        marks.append(pytest.mark.xfail)
    name = f"{case.pdataset.__name__}-{case.name}"
    return pytest.param(case, marks=marks, id=name)


def get_mkdataset_params(no_xfail: bool = False) -> Iterator:
    for case in _params_args:
        if no_xfail:
            case.xfail = False
        yield mkparam(case)


_params_args: list[Case] = [
    Case("trivial", Compositional, dict(attrs=1, vals=1)),
    Case("1_attr", Compositional, dict(attrs=1, vals=3)),
    Case("1", Compositional, dict(attrs=3, vals=3)),
    Case("extra_forms1", Compositional, dict(extra_form_rate=0.1)),
    Case("extra_forms2", Compositional, dict(extra_form_rate=0.4)),
    Case("extra_forms3", Compositional, dict(extra_form_rate=0.6)),
    Case("bom_extra_forms1", Compositional, dict(vals=8, extra_form_rate=0.1)),
    Case("bom_extra_forms2", Compositional, dict(vals=8, extra_form_rate=0.4)),
    Case(
        "bom_extra_forms3",
        Compositional,
        dict(vals=7, extra_form_rate=0.6, samples=500),
    ),
    Case("unbal", Compositional, dict(attrs=3, vals=3, meaning_balanced=False)),
    Case("5_attrs", Compositional, dict(attrs=5, vals=2)),
    Case("5_vals", Compositional, dict(attrs=2, vals=5)),
    Case("5_vals_unbal", Compositional, dict(attrs=2, vals=5, meaning_balanced=False)),
    Case("apf_no1_4", Compositional, dict(atoms_per_form=(1, 4), apf_no_overlap=True)),
    Case("apf_no2", Compositional, dict(atoms_per_form=(2, 2), apf_no_overlap=True)),
    Case("apf_no5", Compositional, dict(atoms_per_form=(5, 5), apf_no_overlap=True)),
    Case(
        "apf1",
        Compositional,
        dict(atoms_per_form=(1, 3), vocab_size=100, shuffle=True),
        tests={"exact": 0.05},
    ),
    Case(
        "apf2",
        Compositional,
        dict(atoms_per_form=(1, 3), vocab_size=100),
        tests={"exact": 0.05},
    ),
    Case(
        "apf3",
        Compositional,
        dict(atoms_per_form=(1, 3), vocab_size=10, shuffle=True, samples=1000),
        tests={"exact": 0.2},
    ),
    Case(
        "apf4",
        Compositional,
        dict(atoms_per_form=(1, 3), vocab_size=10),
        tests={"exact": 0.20},
    ),
    Case(
        "apf5",
        Compositional,
        dict(atoms_per_form=(1, 2), vocab_size=100, attrs=2, vals=2),
        tests=dict(exact=0.05),
    ),
    Case(
        "bom_apf1",
        Compositional,
        dict(vals=8, atoms_per_form=(1, 3), vocab_size=100),
        tests={"exact": 0.03},
    ),
    Case(
        "bom_apf2",
        Compositional,
        dict(vals=8, atoms_per_form=(1, 3), vocab_size=10),
        tests={"exact": 0.7},
    ),
    Case(
        "bom_apf3",
        Compositional,
        dict(vals=8, atoms_per_form=(1, 3), vocab_size=10, shuffle=True),
        tests={"exact": 0.7},
    ),
    Case("syn_2", Compositional, dict(forms_per_meaning=2)),
    Case("syn_many", Compositional, dict(forms_per_meaning=4, samples=500)),
    Case(
        "poly_syn2",
        Compositional,
        dict(forms_per_meaning=3, vals=4, attrs=4, polysemy_rate=0.1, samples=500),
        tests={"exact": 0.8},
    ),
    Case(
        "poly_low",
        Compositional,
        dict(vals=4, attrs=4, polysemy_rate=2 / 16),
        tests={"exact": 0.30},
    ),
    Case(
        "poly_bom",
        Compositional,
        dict(bag_of_meanings=True, vals=10, polysemy_rate=3 / 10, samples=300),
        tests={"exact": 0.5},
        xfail=True,
    ),
    Case(
        "poly_high_bom_unbal",
        Compositional,
        dict(
            bag_of_meanings=True,
            vals=10,
            polysemy_rate=5 / 10,
            meaning_balanced=False,
        ),
        tests={"exact": 0.6},
    ),
    Case(
        "poly_high",
        Compositional,
        # NOTE More data might not be better here
        dict(vals=4, attrs=4, polysemy_rate=3 / 16, shuffle=True),
        tests={"exact": 0.7},
    ),
    Case(
        "poly_high_shuf",
        Compositional,
        dict(vals=4, attrs=4, polysemy_rate=5 / 16, shuffle=True),
        tests={"exact": 0.95},
    ),
    Case(
        "syn_many_small",
        Compositional,
        dict(samples=30, forms_per_meaning=4),
        tests={"exact": 0.99},
    ),
    Case("syn_2_unbal", Compositional, dict(forms_per_meaning=2, syn_balanced=False)),
    Case(
        "syn_many_unbal", Compositional, dict(forms_per_meaning=4, syn_balanced=False)
    ),
    Case("bom", Compositional, dict(vals=8, bag_of_meanings=True)),
    Case(
        "bom_unbal",
        Compositional,
        dict(vals=10, bag_of_meanings=True, meaning_balanced=False),
    ),
    Case(
        "everything",
        Compositional,
        dict(
            meaning_balanced=False,
            forms_per_meaning=4,
            syn_balanced=False,
            samples=500,
        ),
    ),
    Case(
        "everything_bom",
        Compositional,
        dict(
            meaning_balanced=False,
            forms_per_meaning=4,
            syn_balanced=False,
            bag_of_meanings=True,
            samples=500,
        ),
        xfail=True,
    ),
    Case(
        "everything_small",
        Compositional,
        dict(
            meaning_balanced=False, forms_per_meaning=4, syn_balanced=False, samples=30
        ),
        tests={"exact": 0.95},
    ),
    Case(
        "1",
        MixedComp,
        dict(apf_no_overlap=True),
        xfail=True,
        # tests={"exact": 0.95},
    ),
    Case(
        "apf1",
        MixedComp,
        dict(atoms_per_form=(1, 3), vocab_size=10),
        xfail=True,
    ),
    Case(
        "apf2",
        MixedComp,
        dict(atoms_per_form=(1, 3), vocab_size=30),
        xfail=True,
    ),
]
