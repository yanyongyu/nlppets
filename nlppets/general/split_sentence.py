import re
from typing import List, Iterator

SEPARATOR = r"@"
RE_SENTENCE = re.compile(r"(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)", re.UNICODE)
AB_SENIOR = re.compile(r"([A-Z][a-z]{1,2}\.)\s(\w)", re.UNICODE)
AB_ACRONYM = re.compile(r"(\.[a-zA-Z]\.)\s(\w)", re.UNICODE)
UNDO_AB_SENIOR = re.compile(r"([A-Z][a-z]{1,2}\.)" + SEPARATOR + r"(\w)", re.UNICODE)
UNDO_AB_ACRONYM = re.compile(r"(\.[a-zA-Z]\.)" + SEPARATOR + r"(\w)", re.UNICODE)


def _replace_with_separator(text: str, separator: str, regexs: List[re.Pattern]) -> str:
    replacement = r"\1" + separator + r"\2"
    result = text
    for regex in regexs:
        result = regex.sub(replacement, result)
    return result


def split_sentence(text: str) -> Iterator[str]:
    """Split previded text into sentences.

    Args:
        text (str): previded text with `\n` as line separator.

    Yields:
        Iterator[str]: sentences iterator.
    """
    # pre process
    text = re.sub(r"([。！？?])([^”’])", r"\1\n\2", text)
    text = re.sub(r"(\.{6})([^”’])", r"\1\n\2", text)
    text = re.sub(r"(…{2})([^”’])", r"\1\n\2", text)
    text = re.sub(r"([。！？?][”’])([^，。！？?])", r"\1\n\2", text)

    for chunk in text.split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue

        # post process
        processed = _replace_with_separator(chunk, SEPARATOR, [AB_SENIOR, AB_ACRONYM])
        for sentence in RE_SENTENCE.finditer(processed):
            yield _replace_with_separator(
                sentence.group(), r" ", [UNDO_AB_SENIOR, UNDO_AB_ACRONYM]
            )
