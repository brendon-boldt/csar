from pathlib import Path
import typing

import click

from . import analysis as _analysis
from . import scripts


@click.group()
def cli() -> None:
    pass


opt_max_lines = click.option("-l", "--max-lines", default=None, type=int)
opt_inventory_size = click.option("-i", "--max-inventory-size", default=None, type=int)
opt_write_output = click.option("-w", "--write-output", is_flag=True)
arg_input_path = click.argument("input_path", type=Path)
arg_input_paths = click.argument("input_paths", type=Path, nargs=-1)


def _compose(funcs: list[typing.Callable]) -> typing.Callable:
    def f(arg):
        for f in funcs[::-1]:
            arg = f(arg)

    return f

@cli.command()
@arg_input_path
@opt_max_lines
@opt_inventory_size
def induce(**kwargs) -> None:
    """Run morpheme induction arbitrary data."""
    scripts.user_data.main(**kwargs)


@cli.group()
def script():
    pass


script_command = _compose(
    [script.command(), opt_max_lines, opt_inventory_size, opt_write_output]
)


@script_command
def morpho_challenge(**kwargs) -> None:
    scripts.morpho_challenge.main(**kwargs)


@script_command
def mt(**kwargs) -> None:
    scripts.mt.main(**kwargs)


@script_command
@arg_input_path
def amr(**kwargs) -> None:
    scripts.amr.main(**kwargs)


@script_command
def coco(**kwargs) -> None:
    scripts.coco.main(**kwargs)


@script_command
@arg_input_path
def ec_mu(**kwargs) -> None:
    scripts.ec_mu.main(**kwargs)


@script_command
@arg_input_path
def ec(**kwargs) -> None:
    scripts.ec.main(**kwargs)


@cli.command()
@opt_max_lines
@arg_input_path
def ec_vis(input_path: Path, max_lines: int | None) -> None:
    scripts.ec.visualize(input_path, max_lines=max_lines)


@cli.group()
def analysis():
    pass


@analysis.command()
@click.option("-j", "--n-jobs", default=-1, type=int)
@click.option("-n", "--n-samples", default=1, type=int)
@click.option("-o", "--overwrite", is_flag=True)
def collect(**kwargs) -> None:
    _analysis.collect(**kwargs)


@analysis.command()
def report(**kwargs) -> None:
    _analysis.report(**kwargs)


@cli.command()
@arg_input_paths
def read(**kwargs) -> None:
    _analysis.read_dumps(**kwargs)


@cli.command()
def ec_morfessor(**kwargs) -> None:
    scripts.ec_morfessor.main(**kwargs)


cli()
