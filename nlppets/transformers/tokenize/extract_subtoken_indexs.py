from typing import List, Optional

from transformers import PreTrainedTokenizer

from nlppets.general import add_subtoken_symbol


def extract_subtoken_indexs(
    tokenizer: PreTrainedTokenizer,
    input_ids: List[int],
    additional_words: Optional[List[str]] = None,
    sub_symbol: str = "##",
) -> List[int]:
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    if additional_words:
        input_tokens = add_subtoken_symbol(input_tokens, additional_words, sub_symbol)

    return [
        index
        for index, input_token in enumerate(input_tokens)
        if input_token.startswith(sub_symbol)
    ]
