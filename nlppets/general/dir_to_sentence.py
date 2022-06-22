from pathlib import Path
from typing import Iterator

from .file_to_sentence import file_to_sentence as file_to_sentence


def dir_to_sentence(dir: Path) -> Iterator[str]:
    """Recursive load sentences from raw text dir.

    Args:
        dir (Path): provided raw text file directory.

    Yields:
        Iterator[str]: sentences.
    """
    for file in dir.iterdir():
        if file.is_dir():
            yield from dir_to_sentence(file)
        elif file.is_file():
            yield from file_to_sentence(file)
