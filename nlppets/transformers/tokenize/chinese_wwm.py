from typing import Any, Dict, List, Union, Optional

from transformers import PreTrainedTokenizer

from nlppets.general import extract_chinese_token

from .extract_subtoken_indexs import extract_subtoken_indexs


class ChineseWWMTokenizer:
    """Chinese WWM tokenizer."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        *,
        text_column_name: Optional[str] = None,
        padding: Union[str, bool] = False,
        truncation: Union[str, bool] = True,
    ):
        """Tokenizer from transformers should be provided.

        Args:
            tokenizer (PreTrainedTokenizer): transformers pretrained tokenizer.
            max_seq_length (int): max sequence length accepted by model.
            text_column_name (Optional[str], optional): dataset text column name. Defaults to `text`.
            padding (Union[str, bool], optional): whether to add padding or not. Defaults to False.
            truncation (Union[str, bool], optional): whether to truncate the input or not. Defaults to True.
        """
        self.tokenizer = tokenizer

        self.padding = padding
        self.truncation = truncation
        self.max_seq_length = max_seq_length

        self.text_column_name = text_column_name or "text"

    def batched_tokenize_line_by_line(
        self, examples: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """Tokenize texts line by line.

        Can be used with `datasets.Dataset.map`.

        Args:
            examples (Dict[str, List[Any]]): provided texts.

        Returns:
            Dict[str, List[Any]]: dataset dict object.
        """
        encoded_input = self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True,
        )

        # chinese wwm
        batched_input_ids: List[List[int]] = encoded_input["input_ids"]  # type: ignore
        chinese_ref: List[List[int]] = [
            extract_subtoken_indexs(
                self.tokenizer,
                input_ids,
                list(extract_chinese_token(text, min_length=2)),
            )
            for input_ids, text in zip(
                batched_input_ids, examples[self.text_column_name]
            )
        ]

        return {**encoded_input, "chinese_ref": chinese_ref}  # type: ignore

    def batched_tokenize_group_texts(
        self, examples: Dict[str, List[Any]]
    ) -> Dict[str, List[Any]]:
        """Tokenize texts and group them togather.

        Can be used with `datasets.Dataset.map`.

        Args:
            examples (Dict[str, List[Any]]): provided texts.

        Returns:
            Dict[str, List[Any]]: dataset dict object.
        """
        # Concatenate all texts.
        result = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "special_tokens_mask": [],
            "chinese_ref": [],
        }

        tmp_input_ids = []
        tmp_chinese_ref = []
        for text in examples[self.text_column_name]:
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
                    return_special_tokens_mask=True,
                )

                # commit
                result["input_ids"].append(encoded_inputs["input_ids"])
                result["token_type_ids"].append(encoded_inputs["token_type_ids"])
                result["attention_mask"].append(encoded_inputs["attention_mask"])
                result["special_tokens_mask"].append(
                    encoded_inputs["special_tokens_mask"]
                )
                result["chinese_ref"].append(tmp_chinese_ref)

                # reset
                tmp_input_ids = []
                tmp_chinese_ref = []

            tmp_input_ids.extend(input_ids)
            tmp_chinese_ref.extend(
                extract_subtoken_indexs(
                    self.tokenizer,
                    input_ids,
                    list(extract_chinese_token(text, min_length=2)),
                )
            )

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
            result["chinese_ref"].append(tmp_chinese_ref)
        return result
