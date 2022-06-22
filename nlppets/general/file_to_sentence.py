from pathlib import Path
from itertools import chain
from typing import Iterator

from .split_sentence import split_sentence


def file_to_sentence(file: Path) -> Iterator[str]:
    """Read a file and return sentences.

    Note that file content is split into lines first because
    universal newline characters are not supported by split sentence.

    Args:
        file (Path): _description_

    Yields:
        Iterator[str]: _description_
    """
    yield from chain.from_iterable(
        split_sentence(line) for line in file.read_text().splitlines()
    )
