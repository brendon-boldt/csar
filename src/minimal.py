import sys
from pathlib import Path
from unittest.mock import Mock

to_mock = [
    "morfessor",
    "datasets",
    "tokenizers",
    "nltk",
    "Levenshtein",
]

for name in to_mock:
    sys.modules[name] = Mock()

from . import scripts

input_path = Path(sys.argv[1])
scripts.user_data.main(input_path=input_path, max_lines=None, max_inventory_size=None)
