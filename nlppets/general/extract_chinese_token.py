from typing import Iterator

from .is_chinese_token import is_chinese_token
from .split_chinese_token import split_chinese_token


def extract_chinese_token(sentence: str, min_length: int = 1) -> Iterator[str]:
    """Extract chinese tokens with minimum length from a sentence.

    Args:
        sentence (str): sentence str.
        min_length (int, optional): Min length of the token. Defaults to 1.

    Yields:
        Iterator[str]: chinese tokens.
    """
    for token in split_chinese_token(sentence):
        if len(token) >= min_length and is_chinese_token(token):
            yield token
