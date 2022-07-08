from typing import Dict, Type, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOutput as BaseOutput
from transformers.models.bert.modeling_bert import BertIntermediate as BaseIntermediate

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Type[BertPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class BertIntermediate(BaseIntermediate):
    def __init__(self, config: Config):
        super(BertIntermediate, self).__init__(config)

        # added
        self.enhancements = list(config.domain_ffn_enhance.keys())
        for name, size in config.domain_ffn_enhance.items():
            setattr(self, name, nn.Linear(config.hidden_size, size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, H] -> [B, L, I + E]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(BaseOutput):
    def __init__(self, config: Config):
        super(BertOutput, self).__init__(config)

        self.intermediate_size = config.intermediate_size

        # added
        self.enhancements = config.domain_ffn_enhance
        for name, size in config.domain_ffn_enhance.items():
            setattr(self, name, nn.Linear(size, config.hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, I + E] -> [B, L, H]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BERT model to apply feed-forward network domain enhancement.

    Args:
        model (Type[BertPreTrainedModel]): Original BERT model class.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        Type[BertPreTrainedModel]: Patched model class
    """
    intermediate_module: str = (
        "encoder.layer.*.intermediate"
        if model is BertModel
        else "bert.encoder.layer.*.intermediate"
    )
    intermediate_output_module: str = (
        "encoder.layer.*.output"
        if model is BertModel
        else "bert.encoder.layer.*.output"
    )

    origin_init = model.__init__

    def patched_init(
        self: PreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        # patch config if new enhancement provided
        if domain_ffn_enhance is not None:
            config.domain_ffn_enhance = domain_ffn_enhance

        config_with_enhance = cast(Config, config)
        # if domain enhance, replace modules
        if config_with_enhance.domain_ffn_enhance:
            nested_replace_module(
                self,
                intermediate_module,
                lambda _: BertIntermediate(config_with_enhance),
            )
            nested_replace_module(
                self,
                intermediate_output_module,
                lambda _: BertOutput(config_with_enhance),
            )

        self.post_init()

    model.__init__ = patched_init

    return model
