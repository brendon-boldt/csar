from pathlib import Path
import json
import sys
import pickle as pkl
import typing
import difflib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib

from .collect import PATH_F_SCORES
from ..scripts.util import SAVE_DIR as SCRIPTS_SAVE_DIR
from ..common import InductionMorpheme
from ..util import print_morphemes

OUTPUT_DIR = Path("./output")

MODEL_MAP = {
    "csar": "CSAR",
    "ibm1": "IBM Model 1",
    "ibm3": "IBM Model 3",
    "morfessor": "Morfessor",
    "bpe_heuristic": "BPE",
    "unigram_heuristic": "ULM",
    "obs": "Records",
}


def _plot_proc_f1(df: pd.DataFrame, exact: bool) -> None:
    print("=== Plotting F1 scores... ", end="")
    matplotlib.use("pgf")

    # df = df.loc[df["extra_form_rate"] == 0]
    df.loc[df["extra_form_rate"] != 0, ["fuzzy_form_f1", "exact_form_f1"]] = float(
        "nan"
    )
    df = df.drop("dataset", axis=1)
    value_vars = (
        ["exact_form_f1", "exact_f1"] if exact else ["fuzzy_form_f1", "fuzzy_f1"]
    )
    data = (
        df
        # .loc[:, ["fuzzy_form_f1"]]
        .groupby(["model", "dataset"])
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars=["model", "dataset"], value_vars=value_vars)
    )

    model_map = {
        **MODEL_MAP,
        "ibm1": "IBM\nModel 1",
        "ibm3": "IBM\nModel 3",
    }
    data["key"] = data["model"].apply(list(model_map.keys()).index)
    data["model"] = data["model"].replace(model_map)
    data = data.sort_values("key")
    data["metric"] = data["metric"].replace(
        {
            "fuzzy_form_f1": "Form-only",
            "fuzzy_f1": "Form+meaning",
            "exact_form_f1": "Form-only",
            "exact_f1": "Form+meaning",
        }
    )

    sns.set_theme(
        "paper",
        font_scale=0.85,
        style="whitegrid",
        rc={
            # "figure.constrained_layout.use": True,
            "pgf.texsystem": "xelatex",
            # "font.family": "serif",
            # "font.serif": "Computer Modern Roman",
            # "text.usetex": True,
            "pgf.rcfonts": False,
        },
    )
    plot = sns.catplot(
        data,
        y="model",
        x="value",
        hue="metric",
        # alpha=0.5,
        kind="boxen",
        k_depth="full",
        dodge=True,
        aspect=0.6 if exact else 0.68,
        height=3.1,
    )

    plot._legend.remove()  # type: ignore[attr-defined]
    if exact:
        plt.legend(loc="lower right")
        plt.xlim([0.0, 1.00])
    else:
        plt.legend(loc="center left")
        plt.xlim([0.3, 1.00])
    plt.ylabel("Model")
    plt.xlabel("F1-score")

    name = "baselines-" + ("exact" if exact else "fuzzy")
    plot.figure.savefig(f"output/{name}.png")
    plt.close()
    plot.figure.tight_layout(pad=0.7 if exact else -0.1)
    plot.figure.savefig(f"output/{name}.pgf")
    plt.close()
    print("Done. ===")


def _print_proc_f1(df: pd.DataFrame) -> None:
    print("=== F1 Score Summary ===")
    df.loc[
        df["extra_form_rate"] != 0,
        ["fuzzy_form_f1", "fuzzy_form_precision", "fuzzy_form_recall", "exact_form_f1"],
    ] = float("nan")

    metric_map = {
        "exact_form_f1": "Exact $F_1$, form",
        "fuzzy_form_f1": "Fuzzy $F_1$, form",
        "fuzzy_form_precision": "Fuzzy prec., form",
        "fuzzy_form_recall": "Fuzzy recall, form",
        "exact_f1": "Exact $F_1$",
        "fuzzy_f1": "Fuzzy $F_1$",
        "fuzzy_precision": "Fuzzy prec.",
        "fuzzy_recall": "Fuzzy recall",
    }

    table = (
        df.loc[:, list(metric_map)]
        .groupby(["model", "dataset"])
        .mean()
        .groupby("model")
        .mean()
        .loc[list(MODEL_MAP)]  # type: ignore[index]
    )
    table.index.name = None
    table.columns.name = None
    table.rename(index=MODEL_MAP, columns=metric_map, inplace=True)
    print(table)
    table.T.to_latex(OUTPUT_DIR / "proc-table.tex", float_format="%.3f")


def _print_param_effects(df: pd.DataFrame) -> None:
    print("=== Parameter Effects on CSAR ===")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    cols_to_test = [
        "atoms_per_form",
        "bag_of_meanings",
        "extra_form_rate",
        "meaning_balanced",
        "polysemy_rate",
        "samples",
        "shuffle",
        "vocab_size",
        "noncomp",
    ]
    hilos = (
        df.loc[:, cols_to_test]
        .apply(pd.unique)
        .apply(lambda x: pd.Series(x).dropna().tolist())
        .apply(lambda x: sorted(x)[-2:])
    )

    result = pd.DataFrame()

    for col in cols_to_test:
        means = (
            df.groupby(["model", col]).mean(numeric_only=True)
            ["fuzzy_f1"]
        )
        lo, hi = hilos.loc[col]
        res_col = means.xs(hi, level=1) - means.xs(lo, level=1)
        result[col] = res_col
    print(result.loc["csar"])


def _print_compare_csar(df: pd.DataFrame) -> None:
    print("=== Baseline comparison with CSAR ===")
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    df.loc[
        df["extra_form_rate"] != 0,
        ["fuzzy_form_f1", "fuzzy_form_precision", "fuzzy_form_recall"],
    ] = float("nan")

    cols_to_test = [
        "atoms_per_form",
        "bag_of_meanings",
        "extra_form_rate",
        "meaning_balanced",
        "polysemy_rate",
        "samples",
        "shuffle",
        "vocab_size",
        "noncomp",
    ]
    hilos = (
        df.loc[:, cols_to_test]
        .apply(pd.unique)
        .apply(lambda x: pd.Series(x).dropna().tolist())
        .apply(lambda x: sorted(x)[-2:])
    )

    result = pd.DataFrame()

    for col in cols_to_test:
        means = (
            df.groupby(["model", col]).mean(numeric_only=True)["fuzzy_form_f1"]
        )
        lo, hi = hilos.loc[col]
        res_col = means.loc[("csar", lo)] - means.xs(lo, level=1)
        result[col] = res_col
    print(result)


def _procedural_data_reports() -> None:
    raw_df = pd.read_pickle(PATH_F_SCORES)
    cols = list(set(raw_df.columns) - {"metric", "value"})
    pivot = raw_df.pivot(index=cols, columns="metric", values="value")
    df = pivot.reset_index().set_index(["model", "dataset", "seed"])
    df["dataset"] = df.index.get_level_values("dataset")
    df["noncomp"] = df["dataset"].str.contains("MixedComp")

    _print_proc_f1(df)
    _plot_proc_f1(df, True)
    _plot_proc_f1(df, False)
    _print_param_effects(df)
    _print_compare_csar(df)


def report() -> None:
    _procedural_data_reports()
    _ec_data_reports()
    print("Done.")


def read_dumps(input_paths: list[Path]) -> None:
    for p in input_paths:
        _read_dump(p)


def _read_dump(path: Path) -> None:
    is_json = False
    is_pickle = False
    data = None
    try:
        with path.open() as iot:
            data = json.load(iot)
        is_json = True
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    if data is None:
        try:
            with path.open("rb") as iob:
                data = pkl.load(iob)
            is_pickle = True
        except pkl.UnpicklingError:
            raise ValueError(f"Do not know file type of dump {path}.")

    if is_json:
        f = _read_dump_procedural
    elif is_pickle:
        f = _read_dump_inventory
    else:
        raise ValueError()

    print(f"=== Begin {path.stem} ===")
    f(data)
    print(f"=== End {path.stem} ===")


def _read_dump_inventory(data: dict) -> None:
    print_morphemes(data["inventory"])


def _read_dump_procedural(data: dict) -> None:
    def print_metrics(which: str):
        d = data["metrics"][which]
        print(f"{which}:   {d["precision"]:.3f} / {d["recall"]:.3f} / {d["f1"]:.3f}")

    print_metrics("exact")
    print_metrics("fuzzy")

    extracted = sorted(str(x) + "\n" for x in data["morphemes"]["extracted"])
    gtmorph = sorted(str(x) + "\n" for x in data["morphemes"]["ground_truth"])

    sys.stdout.writelines(difflib.unified_diff(gtmorph, extracted))

    print("-- Observerations --")
    obs_strs = sorted(set(str(x) + "\n" for x in data["observations"]))
    sys.stdout.writelines(obs_strs)


def _ec_data_reports() -> None:
    _print_ec_stats()


def _compute_ec_stats(ims: list[InductionMorpheme]) -> dict:
    total_mass = sum(x.prevalence for x in ims)
    form_length = sum(len(x.form) * x.prevalence for x in ims) / total_mass
    meaning_length = sum(len(x.meaning) * x.prevalence for x in ims) / total_mass

    f2m_acc: dict[typing.Any, list[float]] = {}
    m2f_acc: dict[typing.Any, list[float]] = {}
    for im in ims:
        f = im.form
        m = im.meaning
        f2m_acc[f] = f2m_acc.get(f, []) + [im.prevalence]
        m2f_acc[m] = m2f_acc.get(m, []) + [im.prevalence]

    forms_per_meaning = sum(len(xs) * sum(xs) for xs in m2f_acc.values()) / total_mass
    meanings_per_form = sum(len(xs) * sum(xs) for xs in f2m_acc.values()) / total_mass

    def ent(_xs: list[float]) -> float:
        xs = np.array(_xs)
        ps = xs / xs.sum()
        return -(ps * np.log2(ps)).sum()

    synonymy_entropy = sum(ent(xs) * sum(xs) for xs in m2f_acc.values()) / total_mass
    polysemy_entropy = sum(ent(xs) * sum(xs) for xs in f2m_acc.values()) / total_mass
    inventory_size = len(ims)
    inventory_entropy = ent([x.prevalence for x in ims])

    return {
        "form_length": form_length,
        "meaning_length": meaning_length,
        "forms_per_meaning": forms_per_meaning,
        "synonymy_entropy": synonymy_entropy,
        "polysemy_entropy": polysemy_entropy,
        "meanings_per_form": meanings_per_form,
        "inventory_size": inventory_size,
        "inventory_entropy": inventory_entropy,
    }


def _print_ec_stats() -> None:
    raw_data = {}
    for path in SCRIPTS_SAVE_DIR.glob("*.pkl"):
        if not any(path.stem.startswith(x) for x in ["ec-vector", "ec-shapeworld"]):
            continue
        with path.open("rb") as fo:
            data = pkl.load(fo)
            k = path.stem
            raw_data[k] = data

    def mkrecord(d: dict) -> dict:
        return {
            "toposim": d["toposim"],
            **_compute_ec_stats(d["inventory"]),
        }

    df = pd.DataFrame.from_records(
        data=[mkrecord(d) for d in raw_data.values()],
        index=list(raw_data.keys()),
    )

    index_map = {
        "ec-vector-av": "Vector, AV",
        "ec-vector-sparse": "Vector, sparse",
        "ec-shapeworld-ref": "SW, ref",
        "ec-shapeworld-setref": "SW, setref",
        "ec-shapeworld-concept": "SW, concept",
    }
    column_map = {
        "inventory_size": r"$|\text{Inv.}|$",
        "inventory_entropy": r"Inv.\@ $H$",
        "form_length": r"$|\text{Form}|$",
        "meaning_length": r"$|\text{Meaning}|$",
        "synonymy_entropy": "Synonymy",
        "polysemy_entropy": "Polysemy",
        "toposim": "Toposim",
    }
    df.sort_index(inplace=True, key=lambda x: x.map(list(index_map).index))

    df = df[list(column_map)]

    df = df.rename(
        columns=column_map,
        index=index_map,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(df)
    df.to_latex(OUTPUT_DIR / "ec-table.tex", float_format="%.2f")
