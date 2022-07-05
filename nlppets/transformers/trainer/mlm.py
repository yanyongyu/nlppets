from typing import Tuple, Union, Callable, Optional

import torch
from datasets import Dataset
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import TrainOutput, get_last_checkpoint
from transformers import (
    Trainer,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    DataCollatorForWholeWordMask,
    DataCollatorForLanguageModeling,
)


def train_mlm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_args: TrainingArguments,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    *,
    wwm: bool = True,
    compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
    preprocess_logits: Optional[
        Callable[
            [Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor], torch.Tensor
        ]
    ] = None,
):
    if wwm:
        collator = DataCollatorForWholeWordMask(tokenizer=tokenizer)
    else:
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,  # type: ignore
    )

    if train_dataset:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)

        result: TrainOutput = trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if eval_dataset:
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
