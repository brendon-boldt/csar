# CSAR Algorithm for Morpheme Induction

See `./reproduce.sh` for a list of commands to reproduce the results of the paper.

The `pytest` tests are somewhat stale, so don't read too much into their
failure.

## Running morpheme induction

To run the morpheme induction algorithm over arbitrary data, run `python -m src
induce path/to/file.json`.  See `python -m src induce --help` and
`./src/scripts/user_data.py` for configuration of the induction algorithm.
There are two input formats accepted, JSON:

    [
        [0, 1, 2], ["a", "b", "c"],
        [1, 2, 3], ["b", "c", "d"]
    ]

And JSONL:

    [0, 1, 2], ["a", "b", "c"]
    [1, 2, 3], ["b", "c", "d"]

If you do not want to install the whole conda environment, you can use `pip
install -r minimal-requirements.txt` and run `python -m src.minimal
path/to/file.json` and edit options directly in `./src/scripts/user_data.py`.
This requires at least Python 3.12.


## Directory structure

- `reproduce.sh`: script for reproducing the results of paper
- `src/`: Python package, all code
    - `analysis/`: collects data from procedural datasets, builds figures and
      tables
    - `generation/`: generates procedural data
    - `induction/`: implements CSAR and other baseline induction methods
    - `scripts/`: runs induction experiments from paper
    - `__main__.py`: command line interface definition
- `data/`: input data to experiments
- `save/`: intermediate data produced by various commands
- `output/`: tables, figures, TeX files to be included in paper



## Terminology differences

If you are coming from reading the paper, some of the terms in the paper do not
match the codebase at present.  The following list should help resolve those.
- _sparse meanings_: bag-of-meanings, bom
- _multi-token forms_: atoms per form, apf
- _record_: record, observation
