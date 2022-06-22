from typing import List
from pathlib import Path

import datasets
from datasets import (
    Split,
    Value,
    Features,
    Sequence,
    DatasetInfo,
    SplitGenerator,
    DownloadManager,
)

from nlppets.general import dir_to_sentence, extract_chinese_token


class ChineseRawTextDatasetBuilder(datasets.GeneratorBasedBuilder):
    """Simple load sentences from raw text dir."""

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("uint32"),
                "text": Value("string"),
                "chinese_token": Sequence(Value("string")),
            }
        )
        return DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        if not self.config.data_dir:
            raise ValueError("data_dir is not set")

        data_dir = Path(self.config.data_dir)

        return [
            SplitGenerator(
                name=str(Split.TRAIN),
                gen_kwargs={"data_dir": data_dir},
            )
        ]

    def _generate_examples(self, *, data_dir: Path):
        for index, sentence in enumerate(dir_to_sentence(data_dir)):
            yield index, {
                "id": index,
                "text": sentence,
                "chinese_token": extract_chinese_token(sentence, min_length=1),
            }
