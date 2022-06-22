from typing import Iterator

import jieba


def split_chinese_token(sentence: str) -> Iterator[str]:
    """Split token from chinese sentence.

    Note that returned token is not always a chinese word.

    Args:
        sentence (str): sentence str.

    Yields:
        Iterator[str]: sentence tokens.
    """
    yield from jieba.cut(sentence)
