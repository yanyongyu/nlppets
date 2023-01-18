from typing import Any, Dict, Type, Union, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOutput as BaseOutput
from transformers.models.bert.modeling_bert import BertSelfOutput as BaseSelfOutput

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Type[BertPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    hidden_act: Union[str, Any]
    adapters: Dict[str, int]
    """adapters. key for name, value for size."""


class Adapter(nn.Module):
    def __init__(self, config: Config):
        super(Adapter, self).__init__()

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.adapters = config.adapters
        for name, size in config.adapters.items():
            setattr(self, f"{name}_down", nn.Linear(config.hidden_size, size))
            setattr(self, f"{name}_up", nn.Linear(size, config.hidden_size))

    def forward(self, input_states: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, L, A]
        hidden_states = concat_linear(
            *(getattr(self, f"{name}_down") for name in self.adapters)
        )(input_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        # [B, L, A] -> [B, L, H]
        hidden_states = concat_linear(
            *(getattr(self, f"{name}_up") for name in self.adapters)
        )(hidden_states)
        return hidden_states + input_states


class BertSelfOutput(BaseSelfOutput):
    def __init__(self, config):
        super().__init__(config)

        self.adapter = Adapter(config)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput(BaseOutput):
    def __init__(self, config: Config):
        super(BertOutput, self).__init__(config)

        self.adapter = Adapter(config)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def adapter(model: MT, adapters: Optional[Dict[str, int]] = None) -> MT:
    """Modify BERT model to apply feed-forward adapter.

    Args:
        model (Type[BertPreTrainedModel]): Original BERT model class.
        adapters (Optional[Dict[str, int]]):
            Adapters. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        Type[BertPreTrainedModel]: Patched model class
    """
    attention_output_module: str = (
        "encoder.layer.*.attention.output"
        if issubclass(model, BertModel)
        else "bert.encoder.layer.*.attention.output"
    )
    intermediate_output_module: str = (
        "encoder.layer.*.output"
        if issubclass(model, BertModel)
        else "bert.encoder.layer.*.output"
    )

    model = cast(MT, model)

    origin_init = model.__init__

    def patched_init(
        self: BertPreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        # patch config if new enhancement provided
        if adapters is not None:
            config.adapters = adapters

        config_with_enhance = cast(Config, config)
        # if domain enhance, replace modules
        if config_with_enhance.adapters:
            nested_replace_module(
                self,
                attention_output_module,
                lambda _: BertSelfOutput(config_with_enhance),
            )
            nested_replace_module(
                self,
                intermediate_output_module,
                lambda _: BertOutput(config_with_enhance),
            )

        self.post_init()

    model = type(
        f"{model.__name__}_Adapter", (model,), {"__init__": patched_init}
    )  # type: ignore

    return model
