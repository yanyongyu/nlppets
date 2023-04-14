from typing import List, Tuple, Union, Callable, Optional

import torch
from datasets import Dataset
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import TrainOutput, get_last_checkpoint
from transformers import (
    Trainer,
    EvalPrediction,
    PreTrainedModel,
    TrainerCallback,
    DefaultDataCollator,
    PreTrainedTokenizerBase,
)


def train_clm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingArguments,
    train_dataset: Optional[Dataset] = None,
    eval_dataset: Optional[Dataset] = None,
    *,
    collator: Optional[Callable] = None,
    compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    preprocess_logits: Optional[
        Callable[
            [Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor], torch.Tensor
        ]
    ] = None,
):
    """Causal Language Model training function.

    Args:
        model (PreTrainedModel): The model to train.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        training_args (TrainingArguments): The training arguments.
        train_dataset (Optional[Dataset], optional): Training dataset to use.
        eval_dataset (Optional[Dataset], optional): Evaluation dataset to use.
        collator (Optional[Callable], optional): Data collator to use. Defaults to None.
        compute_metrics (Optional[Callable[[EvalPrediction], dict]], optional):
            Metrics function. Defaults to None.
        callbacks (Optional[List[TrainerCallback]], optional): Callbacks to use. Defaults to None.
        preprocess_logits (Optional[Callable[[Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor], torch.Tensor]], optional):
            Logits preprocess function. Defaults to None.
    """
    collator = collator or DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        preprocess_logits_for_metrics=preprocess_logits,  # type: ignore
    )

    if train_dataset:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

        result: TrainOutput = trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()

    if eval_dataset:
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
