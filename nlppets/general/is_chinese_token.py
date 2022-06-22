def is_chinese_token(token: str) -> bool:
    """Check if token str is Chinese.

    Args:
        token (str): provided token str.

    Returns:
        bool: token is Chinese or not.
    """
    return any(
        (0x3400 <= ord(char) <= 0x4DBF)
        or (0x4E00 <= ord(char) <= 0x9FFF)
        or (0xF900 <= ord(char) <= 0xFAFF)
        or (0x20000 <= ord(char) <= 0x2A6DF)
        or (0x2A700 <= ord(char) <= 0x2B73F)
        or (0x2B740 <= ord(char) <= 0x2B81F)
        or (0x2B820 <= ord(char) <= 0x2CEAF)
        or (0x2F800 <= ord(char) <= 0x2FA1F)
        for char in token
    )
