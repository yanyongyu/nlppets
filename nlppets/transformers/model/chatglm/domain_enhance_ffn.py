import inspect
from functools import wraps
from typing import Dict, Type, TypeVar, Callable, Optional, Protocol, cast

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from nlppets.general import MonkeyPatch
from nlppets.torch import concat_linear

MT = TypeVar("MT", bound=Type[PreTrainedModel])


class Config(Protocol):
    hidden_size: int
    intermediate_size: int
    domain_ffn_enhance: Dict[str, int]
    """domain pre-training enhancements. key for name, value for size."""


def get_patched_module(origin_mlp: Type[nn.Module], config: Config) -> Type[nn.Module]:
    class ChatGLMMLP(nn.Module):
        enhancements: Dict[str, int]
        hidden_size: int
        inner_hidden_size: int
        dense_h_to_4h: torch.nn.Linear
        dense_4h_to_h: torch.nn.Linear
        activation_func: Callable[[torch.Tensor], torch.Tensor]

        def __init__(self, *args, **kwargs) -> None:
            origin_mlp.__init__(self, *args, **kwargs)

            dtype = self.dense_h_to_4h.weight.dtype
            self.enhancements = config.domain_ffn_enhance
            for name, size in config.domain_ffn_enhance.items():
                setattr(
                    self, f"{name}_up", nn.Linear(config.hidden_size, size, dtype=dtype)
                )
                setattr(
                    self,
                    f"{name}_down",
                    nn.Linear(size, config.hidden_size, dtype=dtype),
                )

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

    return ChatGLMMLP


def domain_enhance_ffn(
    model: MT, domain_ffn_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify ChatGLM model to apply feed-forward network domain enhancement.

    Args:
        model (Type[ChatGLMPreTrainedModel]): Original ChatGLM model.
        domain_ffn_enhance (Optional[Dict[str, int]]):
            Domain enhancements. key for name, value for size.
            If None is provided, will read from existing configs.

    Returns:
        Type[ChatGLMPreTrainedModel]: Patched model
    """

    model = cast(MT, model)

    origin_init = model.__init__
    chatglm_module = inspect.getmodule(model)
    origin_mlp = getattr(chatglm_module, "GLU")

    def patched_init(self: PreTrainedModel, config: PretrainedConfig, *args, **kwargs):
        # patch config if new enhancement provided
        if domain_ffn_enhance is not None:
            config.domain_ffn_enhance = domain_ffn_enhance

        config_with_enhance = cast(Config, config)

        with MonkeyPatch.context() as m:
            m.setattr(
                chatglm_module,
                "GLU",
                get_patched_module(origin_mlp, config_with_enhance),
            )

            origin_init(self, config, *args, **kwargs)

    return wraps(
        model,
        assigned=(
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
            "__annotations__",
        ),
        updated=(),
    )(
        type(f"{model.__name__}_EnhanceFFN", (model,), {"__init__": patched_init})
    )  # type: ignore
