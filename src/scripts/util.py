from pathlib import Path
import pickle as pkl

from ..common import InductionMorpheme
from ..induction import CsarPipeline
from ..util import compute_toposim, print_morphemes

OUTPUT_DIR = Path("./output/scripts")
SAVE_DIR = Path("./save/scripts")


def save_output(
    name: str,
    inventory: list[InductionMorpheme],
    *,
    extra_data: dict | None = None,
    no_form_space: bool = False,
    tex_line_limit: None | int = None,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # Save pickle file
    save_data = dict(
        inventory=inventory,
        **(extra_data or {}),
    )
    with (SAVE_DIR / f"{name}.pkl").open("wb") as bfo:
        pkl.dump(save_data, bfo)

    # Save TeX file
    def _str(_x: object) -> str:
        x = str(_x)
        match x:
            case "^":
                return "\\^{}"
            case "$" | "_":
                return "\\" + x
            case _:
                return x

    form_join_str = "" if no_form_space else " "

    with (OUTPUT_DIR / f"{name}.tex").open("w") as fo:
        for im in inventory[:tex_line_limit]:
            f, m = im.to_morpheme()
            meaning_str = ", ".join("".join(_str(y) for y in str(x)) for x in m)
            form_str = form_join_str.join(_str(x) for x in f)
            line = f"(``{form_str}'', \\{{{meaning_str}\\}})\n"
            fo.write(line)


def run(
    name: str,
    pipeline: CsarPipeline,
    *,
    write_output: bool,
    do_toposim: bool = False,
    no_form_space: bool = False,
) -> None:
    extra_data = {}
    pipeline.extractor.show_progress = True
    pipeline.induce()
    if not write_output:
        print_morphemes(pipeline=pipeline)
    if do_toposim:
        extra_data["toposim"] = compute_toposim(pipeline.observations)
    if write_output:
        save_output(
            name,
            pipeline.extractor.induction_morphemes,
            tex_line_limit=100,
            no_form_space=no_form_space,
            extra_data=extra_data,
        )
