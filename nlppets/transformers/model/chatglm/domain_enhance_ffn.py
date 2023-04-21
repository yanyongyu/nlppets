from typing import Dict, Type, TypeVar, Callable, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=PreTrainedModel)


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


class ChatGLMMLP:
    enhancements: Dict[str, int]
    hidden_size: int
    inner_hidden_size: int
    dense_h_to_4h: torch.nn.Linear
    dense_4h_to_h: torch.nn.Linear
    activation_func: Callable[[torch.Tensor], torch.Tensor]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # patched for domain enhancements
        # [L, B, H] -> [L, B, I + E]
        hidden_states = concat_linear(
            self.dense_h_to_4h,
            *(getattr(self, f"{name}_up") for name in self.enhancements.keys()),
        )(hidden_states)
        hidden_states = self.activation_func(hidden_states)

        # patched for domain enhancements
        # [L, B, I + E] -> [L, B, H]
        return concat_linear(
            self.dense_4h_to_h,
            *(getattr(self, f"{name}_down") for name in self.enhancements.keys()),
        )(hidden_states)


def _patch_module(module: ChatGLMMLP, config: Config) -> None:
    dtype = module.dense_h_to_4h.weight.dtype

    module.enhancements = config.domain_ffn_enhance
    for name, size in config.domain_ffn_enhance.items():
        setattr(module, f"{name}_up", nn.Linear(config.hidden_size, size, dtype=dtype))
        setattr(
            module, f"{name}_down", nn.Linear(size, config.hidden_size, dtype=dtype)
        )

    module.forward = ChatGLMMLP.forward.__get__(module, ChatGLMMLP)


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify ChatGLM model to apply feed-forward network domain enhancement.

    Args:
        model (ChatGLMPreTrainedModel): Original ChatGLM model.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        ChatGLMPreTrainedModel: Patched model
    """
    mlp_module: str = (
        "transformer.layers.*.mlp" if hasattr(model, "transformer") else "layers.*.mlp"
    )

    # patch config if new enhancement provided
    if domain_ffn_enhance is not None:
        model.config.domain_ffn_enhance = domain_ffn_enhance

    config_with_enhance = cast(Config, model.config)
    # if domain enhance, replace modules
    if config_with_enhance.domain_ffn_enhance:
        nested_replace_module(
            model,
            mlp_module,
            lambda _, module: _patch_module(module, config_with_enhance),
        )

    return model
