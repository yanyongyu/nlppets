from typing import Dict, Type, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.bloom.modeling_bloom import dropout_add
from transformers.models.bloom.modeling_bloom import BloomMLP as BaseMLP
from transformers.models.bloom import BloomModel, BloomConfig, BloomPreTrainedModel

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Type[BloomPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class BloomMLP(BaseMLP):
    def __init__(self, config: Config):
        super(BloomMLP, self).__init__(cast(BloomConfig, config))

        # added
        self.enhancements = config.domain_ffn_enhance
        for name, size in config.domain_ffn_enhance.items():
            setattr(self, f"{name}_up", nn.Linear(config.hidden_size, size))
            setattr(self, f"{name}_down", nn.Linear(size, config.hidden_size))

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


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BLOOM model to apply feed-forward network domain enhancement.

    Args:
        model (Type[BloomPreTrainedModel]): Original BLOOM model class.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        Type[BloomPreTrainedModel]: Patched model class
    """
    mlp_module: str = (
        "h.*.mlp" if issubclass(model, BloomModel) else "transformer.h.*.mlp"
    )

    model = cast(MT, model)

    origin_init = model.__init__

    def patched_init(
        self: BloomPreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
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
                mlp_module,
                lambda _: BloomMLP(config_with_enhance),
            )

        self.post_init()

    model = type(
        f"{model.__name__}_EnhanceFFN", (model,), {"__init__": patched_init}
    )  # type: ignore

    return model
