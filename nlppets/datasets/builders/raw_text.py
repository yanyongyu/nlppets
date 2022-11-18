from typing import List
from pathlib import Path
from dataclasses import field, dataclass

import datasets
from datasets import (
    Split,
    Value,
    Features,
    DatasetInfo,
    SplitGenerator,
    DownloadManager,
)

from nlppets.general import dir_to_sentence, file_to_sentence


@dataclass
class RawTextDatasetConfig(datasets.BuilderConfig):
    dirs: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


class RawTextDatasetBuilder(datasets.GeneratorBasedBuilder):
    """Simple loading sentences from raw text dirs/files.

    Examples:
        >>> builder = RawTextDatasetBuilder(dirs=["./data/raw_text/"], files=["./data/text.txt"])
        >>> builder.download_and_prepare()
        >>> dataset = builder.as_dataset()
    """

    config: RawTextDatasetConfig
    BUILDER_CONFIG_CLASS = RawTextDatasetConfig

    def _info(self) -> DatasetInfo:
        features = Features(
            {
                "id": Value("uint32"),
                "text": Value("string"),
            }
        )
        return DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[SplitGenerator]:
        dirs = tuple(Path(d) for d in self.config.dirs)
        files = tuple(Path(f) for f in self.config.files)

        return [
            SplitGenerator(
                name=str(Split.TRAIN),
                gen_kwargs={"dirs": dirs, "files": files},
            )
        ]

    def _generate_examples(self, *, dirs: List[Path], files: List[Path]):
        index = -1
        for dir in dirs:
            for sentence in dir_to_sentence(dir):
                index += 1
                yield index, {"id": index, "text": sentence}
        for file in files:
            for sentence in file_to_sentence(file):
                index += 1
                yield index, {"id": index, "text": sentence}
