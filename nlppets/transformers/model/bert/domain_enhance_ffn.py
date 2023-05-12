import inspect
from typing import Dict, Type, Union, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOutput as BaseOutput
from transformers.models.bert.modeling_bert import BertIntermediate as BaseIntermediate

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Union[BertPreTrainedModel, Type[BertPreTrainedModel]])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class BertIntermediate(BaseIntermediate):
    enhancements: Dict[str, int]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, H] -> [B, L, I + E]
        hidden_states = concat_linear(
            self.dense, *(getattr(self, name) for name in self.enhancements)
        )(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(BaseOutput):
    enhancements: Dict[str, int]

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


def _patch_bert_intermediate(module: BertIntermediate, config: Config):
    module.enhancements = config.domain_ffn_enhance
    for name, size in config.domain_ffn_enhance.items():
        setattr(module, name, nn.Linear(config.hidden_size, size))
    module.forward = BertIntermediate.forward.__get__(module, module.__class__)


def _patch_bert_output(module: BertOutput, config: Config):
    module.enhancements = config.domain_ffn_enhance
    for name, size in config.domain_ffn_enhance.items():
        setattr(module, name, nn.Linear(size, config.hidden_size))
    module.forward = BertOutput.forward.__get__(module, module.__class__)


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BERT model to apply feed-forward network domain enhancement.

    Args:
        model (BertPreTrainedModel| Type[BertPreTrainedModel]): Original BERT model class.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        BertPreTrainedModel| Type[BertPreTrainedModel]: Patched model class
    """
    intermediate_module: str = (
        "encoder.layer.*.intermediate"
        if isinstance(model, BertModel)
        or (inspect.isclass(model) and issubclass(model, BertModel))
        else "bert.encoder.layer.*.intermediate"
    )
    intermediate_output_module: str = (
        "encoder.layer.*.output"
        if isinstance(model, BertModel)
        or (inspect.isclass(model) and issubclass(model, BertModel))
        else "bert.encoder.layer.*.output"
    )

    def _patch_model(m: BertPreTrainedModel):
        # patch config if new enhancement provided
        if domain_ffn_enhance is not None:
            m.config.domain_ffn_enhance = domain_ffn_enhance

        config_with_enhance = cast(Config, m.config)
        # if domain enhance, replace modules
        if config_with_enhance.domain_ffn_enhance:
            nested_replace_module(
                m,
                intermediate_module,
                lambda _, module: _patch_bert_intermediate(module, config_with_enhance),
            )
            nested_replace_module(
                m,
                intermediate_output_module,
                lambda _, module: _patch_bert_output(module, config_with_enhance),
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

    mc = type(f"{mc.__name__}_EnhanceFFN", (mc,), {"__init__": patched_init})

    return mc  # type: ignore
