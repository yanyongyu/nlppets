from typing import List


def add_subtoken_symbol(
    input_tokens: List[str], additional_words: List[str], sub_symbol: str = "##"
) -> List[str]:
    # return a copy if no additional words
    if not additional_words:
        return input_tokens.copy()

    # remove all subtoken symbols for join tokens
    cleaned_tokens = [
        token[len(sub_symbol) :] if token.startswith(sub_symbol) else token
        for token in input_tokens
    ]
    new_tokens = []

    max_word_len = max(len(w) for w in additional_words)

    # iter the provided tokens
    start, end = 0, len(input_tokens)
    while start < end:
        # commit current token first
        current_token = input_tokens[start]
        new_tokens.append(current_token)
        # continue if the current token is not a subtoken
        if not current_token.startswith(sub_symbol):
            # get max length to find
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                # join tokens to word using cleaned tokens
                whole_word = "".join(cleaned_tokens[start : start + i])
                # found in additional words
                if whole_word in additional_words:
                    # commit subtokens
                    new_tokens.extend(
                        sub_symbol + cleaned_tokens[j]
                        for j in range(start + 1, start + i)
                    )
                    # change start index to last subtoken
                    start += i - 1
                    break

        # next token
        start += 1
    return new_tokens
