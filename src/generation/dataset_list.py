import typing
import itertools


from . import procedural

PD = typing.TypeVar("PD", bound=procedural.ProceduralDataset)


class DatasetArgs(typing.NamedTuple, typing.Generic[PD]):
    name: str
    pdataset: type[PD]
    pd_args: dict


A = DatasetArgs
C = procedural.Compositional
MC = procedural.MixedComp


dataset_args: list[DatasetArgs] = [
    A("base", C, dict()),
]


def _dict_to_name(kwargs: dict[str, typing.Any]) -> str:
    items = sorted(kwargs.items())

    def to_str(k, v) -> str:
        match k:
            case "atoms_per_form":
                return "" if v[1] == 1 else f"apf={v[0]}-{v[1]}"
            case "forms_per_meaning":
                return "" if v == 1 else f"fpm={v}"
            case "vocab_size":
                return f"vs={v}"
            case "polysemy_rate":
                return "" if v == 0 else f"poly={v}"
            case "samples":
                return f"samps={v}"
            case "shuffle":
                return "" if v == 0 else "shuf"
            case "bag_of_meanings":
                return "" if v == 0 else "bom"
            case "vals":
                return f"vals={v}"
            case "meaning_balanced":
                return "" if v == 1 else "unbal"
            case "extra_form_rate":
                return "" if v == 0 else f"efr-{v}"
            case _:
                raise ValueError(k)

    strs = [to_str(k, v) for k, v in items]
    return "_".join(s for s in strs if s)

    return str(items)


def get_dataset_args() -> typing.Iterator[DatasetArgs]:
    default_kwargs = {
        "attrs": 4,
        "vals": 4,
    }
    grid_args: list[list[dict]] = [
        [{"forms_per_meaning": x} for x in (1, 3)],
        [
            {"bag_of_meanings": False},
            {"bag_of_meanings": True, "vals": 8},
        ],
        [
            {"atoms_per_form": (1, 1)},
            {"atoms_per_form": (1, 4), "vocab_size": 10},
            {"atoms_per_form": (1, 4), "vocab_size": 50},
        ],
        [{"polysemy_rate": x} for x in (0, 0.15)],
        [{"shuffle": x} for x in (True, False)],
        [{"meaning_balanced": x} for x in (True, False)],
        [{"extra_form_rate": x} for x in (0, 0.5)],
        [{"samples": x} for x in (50, 500)],
        [{"_compositional": x} for x in (True, False)],
    ]

    for _kwargs in itertools.product(*grid_args):
        kwargs = {k: v for d in _kwargs for k, v in d.items()}

        _compositional = kwargs.pop("_compositional")
        _class = procedural.Compositional if _compositional else procedural.MixedComp

        if not _compositional:
            match kwargs["forms_per_meaning"]:
                case 1:
                    del kwargs["forms_per_meaning"]
                case _:
                    continue
            match kwargs["polysemy_rate"]:
                case 0:
                    del kwargs["polysemy_rate"]
                case _:
                    continue

        name = _class.__name__ + "_" + _dict_to_name(kwargs)

        kwargs = default_kwargs | kwargs

        yield DatasetArgs(name, _class, kwargs)


if __name__ == "__main__":
    for x in get_dataset_args():
        print(x.name)
        ds = x.pdataset(seed=0, **x.pd_args)
        ds.get_observations()
