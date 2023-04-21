import inspect
from typing import Dict, Type, Union, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.bloom.modeling_bloom import dropout_add
from transformers.models.bloom.modeling_bloom import BloomMLP as BaseMLP
from transformers.models.bloom import BloomModel, BloomConfig, BloomPreTrainedModel

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Union[BloomPreTrainedModel, Type[BloomPreTrainedModel]])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class BloomMLP(BaseMLP):
    enhancements: Dict[str, int]

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        # patched for domain enhancements
        # [B, L, H] -> [B, L, I + E]
        hidden_states = concat_linear(
            self.dense_h_to_4h,
            *(getattr(self, f"{name}_up") for name in self.enhancements.keys()),
        )(hidden_states)
        hidden_states = self.gelu_impl(hidden_states)

        # patched for domain enhancements
        # [B, L, I + E] -> [B, L, H]
        hidden_states = concat_linear(
            self.dense_4h_to_h,
            *(getattr(self, f"{name}_down") for name in self.enhancements.keys()),
        )(hidden_states)

        return dropout_add(hidden_states, residual, self.hidden_dropout, self.training)


def _patch_module(module: BloomMLP, config: Config) -> None:
    module.enhancements = config.domain_ffn_enhance
    for name, size in config.domain_ffn_enhance.items():
        setattr(module, f"{name}_up", nn.Linear(config.hidden_size, size))
        setattr(module, f"{name}_down", nn.Linear(size, config.hidden_size))

    module.forward = BloomMLP.forward.__get__(module, BloomMLP)


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BLOOM model to apply feed-forward network domain enhancement.

    Args:
        model (BloomPreTrainedModel | Type[BloomPreTrainedModel]): Original BLOOM model class.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        BloomPreTrainedModel | Type[BloomPreTrainedModel]: Patched model class
    """
    mlp_module: str = (
        "h.*.mlp"
        if isinstance(model, BloomModel)
        or (inspect.isclass(model) and issubclass(model, BloomModel))
        else "transformer.h.*.mlp"
    )

    def _patch_model(m: BloomPreTrainedModel):
        # patch config if new enhancement provided
        if domain_ffn_enhance is not None:
            m.config.domain_ffn_enhance = domain_ffn_enhance

        config_with_enhance = cast(Config, m.config)
        # if domain enhance, replace modules
        if config_with_enhance.domain_ffn_enhance:
            nested_replace_module(
                m,
                mlp_module,
                lambda _, module: _patch_module(module, config_with_enhance),
            )

    if not inspect.isclass(model):
        m = cast(BloomPreTrainedModel, model)
        _patch_model(m)
        return m  # type: ignore

    mc = cast(Type[BloomPreTrainedModel], model)

    origin_init = mc.__init__

    def patched_init(
        self: BloomPreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        _patch_model(self)

        self.post_init()

    mc = type(f"{mc.__name__}_EnhanceFFN", (mc,), {"__init__": patched_init})

    return mc  # type: ignore
