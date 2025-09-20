import tempfile
import ast
import typing
import os
import sys

import morfessor

from .pipeline import Pipeline
from .. import common
from .. import logutils



class MorfessorPipeline(Pipeline):
    def induce(self) -> list[common.Morpheme]:
        morfessor._logger.setLevel(logutils.logging.WARNING)
        parser = morfessor.get_default_argparser()

        def temp(mode: str):
            return tempfile.NamedTemporaryFile(mode, delete_on_close=False)

        form_constructor: typing.Any = None

        with temp("w+") as input_fo, temp("r") as output_fo:
            for utt, _ in self.observations:
                if form_constructor is None:
                    form_constructor = type(utt[0])
                line = ",".join(str(x) for x in utt)
                input_fo.write(line)
                input_fo.write("\n")
            input_fo.flush()

            input_fo.seek(0)
            # print(input_fo.read())

            # fmt: off
            argv = [
                "-t", input_fo.name,
                "-T", input_fo.name,
                "--atom-separator", ",",
                "--output-format-separator", ";",
                "-R", "1",
                "-x", output_fo.name,
                "-o", "/dev/null",
            ]
            # fmt: on
            args = parser.parse_args(argv)
            with open(os.devnull, "w") as dev_null:
                _stdout = sys.stdout
                _stderr = sys.stderr
                sys.stdout = dev_null
                sys.stderr = dev_null
                morfessor.main(args)
                sys.stdout = _stdout
                sys.stderr = _stderr
            output_str = output_fo.read()

        morphemes: list = []
        for line in output_str.split("\n"):
            if not line:
                continue
            raw_morph = line[line.index(" ") + 1 :]
            evaled = ast.literal_eval(raw_morph)
            mapped = tuple(form_constructor(x) for x in evaled)
            morphemes.append((mapped, frozenset()))
        self.morphemes = morphemes

        return self.get_morphemes()

    def get_morphemes(self) -> list[common.Morpheme]:
        return self.morphemes
