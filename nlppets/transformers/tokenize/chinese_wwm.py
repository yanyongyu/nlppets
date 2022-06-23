from typing import Any, Dict, List, Union, Optional

from transformers import PreTrainedTokenizer

from nlppets.general import is_chinese_token


def _add_sub_symbol(input_tokens: List[str], chinese_tokens: List[str]) -> List[str]:
    if not chinese_tokens:
        return input_tokens

    max_word_len = max(len(w) for w in chinese_tokens)
    start, end = 0, len(input_tokens)
    while start < end:
        single_word = True
        if is_chinese_token(input_tokens[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(input_tokens[start : start + i])
                if whole_word in chinese_tokens:
                    for j in range(start + 1, start + i):
                        input_tokens[j] = (
                            input_tokens[j]
                            if input_tokens[j].startswith("##")
                            else f"##{input_tokens[j]}"
                        )
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return input_tokens


class ChineseWWMTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        *,
        text_column_name: Optional[str] = None,
        chinese_token_column_name: Optional[str] = None,
        padding: Union[str, bool] = False,
        truncation: Union[str, bool] = True,
    ):
        self.tokenizer = tokenizer

        self.padding = padding
        self.truncation = truncation
        self.max_seq_length = max_seq_length

        self.text_column_name = text_column_name or "text"
        self.chinese_token_column_name = chinese_token_column_name or "chinese_token"

    def batched_tokenize_line_by_line(self, examples: Dict[str, List[Any]]):
        encoded_input = self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )

        # chinese token not provided
        if self.chinese_token_column_name not in examples:
            return encoded_input

        # chinese wwm
        chinese_ref: List[List[int]] = []
        for input_ids, chinese_token in zip(
            encoded_input["input_ids"], examples[self.chinese_token_column_name]  # type: ignore
        ):
            input_tokens = [self.tokenizer._convert_id_to_token(i) for i in input_ids]
            input_tokens = _add_sub_symbol(input_tokens, chinese_token)
            refs = [
                index
                for index, input_token in enumerate(input_tokens)
                if input_token.startswith("##")
            ]
            chinese_ref.append(refs)
        return {**encoded_input, "chinese_ref": chinese_ref}

    def batched_tokenize_group_texts(self, examples: Dict[str, List[Any]]):
        has_chinese_ref = self.chinese_token_column_name in examples
        # Concatenate all texts.
        result = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "special_tokens_mask": [],
        }
        if has_chinese_ref:
            result["chinese_ref"] = []

        tmp_input_ids = []
        tmp_chinese_ref = []
        for index, text in enumerate(examples[self.text_column_name]):
            input_ids = self.tokenizer.encode(text)

            # overflow, commit first
            new_length = len(tmp_input_ids) + len(input_ids)
            if new_length > self.max_seq_length and len(tmp_input_ids) > 0:
                # pad and truncate
                encoded_inputs = self.tokenizer.prepare_for_model(
                    tmp_input_ids,
                    padding=self.padding,
                    truncation=self.truncation,
                    max_length=self.max_seq_length,
                    # We use this option because DataCollatorForLanguageModeling
                    # is more efficient when it receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

                # commit
                result["input_ids"].append(encoded_inputs["input_ids"])
                result["token_type_ids"].append(encoded_inputs["token_type_ids"])
                result["attention_mask"].append(encoded_inputs["attention_mask"])
                result["special_tokens_mask"].append(
                    encoded_inputs["special_tokens_mask"]
                )
                if has_chinese_ref:
                    result["chinese_ref"].append(tmp_chinese_ref)

                # reset
                tmp_input_ids = []
                tmp_chinese_ref = []

            input_tokens = [self.tokenizer._convert_id_to_token(i) for i in input_ids]
            input_tokens = _add_sub_symbol(
                input_tokens, examples[self.chinese_token_column_name][index]
            )
            refs = [
                len(tmp_input_ids) + index
                for index, input_token in enumerate(input_tokens)
                if input_token.startswith("##")
            ]

            tmp_input_ids.extend(input_ids)
            tmp_chinese_ref.extend(refs)

        # commit last one
        if tmp_input_ids:
            # pad and truncate
            encoded_inputs = self.tokenizer.prepare_for_model(
                tmp_input_ids,
                padding=self.padding,
                truncation=True,
                max_length=self.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling
                # is more efficient when it receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

            # commit
            result["input_ids"].append(encoded_inputs["input_ids"])
            result["token_type_ids"].append(encoded_inputs["token_type_ids"])
            result["attention_mask"].append(encoded_inputs["attention_mask"])
            result["special_tokens_mask"].append(encoded_inputs["special_tokens_mask"])
            if has_chinese_ref:
                result["chinese_ref"].append(tmp_chinese_ref)
        return result
