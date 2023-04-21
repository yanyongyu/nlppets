import inspect
from typing import Any, Dict, Type, Union, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOutput as BaseOutput
from transformers.models.bert.modeling_bert import BertSelfOutput as BaseSelfOutput

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Union[BertPreTrainedModel, Type[BertPreTrainedModel]])


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
    adapter: Adapter

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertOutput(BaseOutput):
    adapter: Adapter

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def _patch_self_output(module: BertSelfOutput, config: Config) -> None:
    module.adapter = Adapter(config)
    module.forward = BertSelfOutput.forward.__get__(module)


def _patch_output(module: BertOutput, config: Config) -> None:
    module.adapter = Adapter(config)
    module.forward = BertOutput.forward.__get__(module)


def adapter(model: MT, adapters: Optional[Dict[str, int]] = None) -> MT:
    """Modify BERT model to apply feed-forward adapter.

    Args:
        model (BertPreTrainedModel, Type[BertPreTrainedModel]): Original BERT model class.
        adapters (Optional[Dict[str, int]]):
            Adapters. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        BertPreTrainedModel, Type[BertPreTrainedModel]: Patched model class
    """
    attention_output_module: str = (
        "encoder.layer.*.attention.output"
        if isinstance(model, BertModel)
        or (inspect.isclass(model) and issubclass(model, BertModel))
        else "bert.encoder.layer.*.attention.output"
    )
    intermediate_output_module: str = (
        "encoder.layer.*.output"
        if isinstance(model, BertModel)
        or (inspect.isclass(model) and issubclass(model, BertModel))
        else "bert.encoder.layer.*.output"
    )

    def _patch_model(m: BertPreTrainedModel):
        # patch config if new enhancement provided
        if adapters is not None:
            m.config.adapters = adapters

        config_with_enhance = cast(Config, m.config)
        # if domain enhance, replace modules
        if config_with_enhance.adapters:
            nested_replace_module(
                m,
                attention_output_module,
                lambda _, module: _patch_self_output(module, config_with_enhance),
            )
            nested_replace_module(
                m,
                intermediate_output_module,
                lambda _, module: _patch_output(module, config_with_enhance),
            )

    if not inspect.isclass(model):
        m = cast(BertPreTrainedModel, model)
        _patch_model(m)
        return m  # type: ignore

    mc = cast(Type[BertPreTrainedModel], model)

    origin_init = mc.__init__

    def patched_init(
        self: BertPreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        _patch_model(self)

        self.post_init()

    mc = type(f"{mc.__name__}_Adapter", (mc,), {"__init__": patched_init})

    return mc  # type: ignore
