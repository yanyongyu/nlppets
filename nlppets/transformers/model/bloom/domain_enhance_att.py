import math
from typing import Any, Dict, Type, Tuple, Literal, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.models.bloom.modeling_bloom import dropout_add
from transformers.models.bloom import BloomModel, BloomConfig, BloomPreTrainedModel
from transformers.models.bloom.modeling_bloom import BloomAttention as BaseAttention

from nlppets.torch import concat_linear, nested_replace_module

MT = TypeVar("MT", bound=Type[BloomPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    num_attention_heads: int
    domain_att_enhance: Dict[str, int]
    """domain pre-training enhancements."""


class BloomAttention(BaseAttention):
    def __init__(self, config: Config):
        super(BloomAttention, self).__init__(cast(BloomConfig, config))

        # added
        self.enhancements = config.domain_att_enhance
        self.additional_heads = sum(config.domain_att_enhance.values())
        for name, size in config.domain_att_enhance.items():
            setattr(
                self,
                f"{name}",
                nn.Linear(config.hidden_size, self.head_dim * size * 3),
            )

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = fused_qkv.shape
        fused_qkv = fused_qkv.view(
            batch_size,
            seq_length,
            self.num_heads + self.additional_heads,
            3,
            self.head_dim,
        )
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        num_heads = self.num_heads + self.additional_heads
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // num_heads

        # First view to decompose the batch size
        # [B * (HN + EN), L, HS] -> [B, (HN + EN), L, HS]
        x = x.view(batch_size, num_heads, seq_length, self.head_dim)

        # [B, (HN + EN), L, HS] -> [B, L, (HN + EN), HS]
        x = x.permute(0, 2, 1, 3)

        # [B, L, (HN + EN), HS] -> [B, L, (HN + EN) * HS]
        return x.reshape(batch_size, seq_length, num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # [B, L, H] -> [B, L, (HN + EN) * 3 * HS]
        fused_qkv = concat_linear(
            self.query_key_value,
            *(getattr(self, name) for name in self.enhancements.keys()),
        )(hidden_states)

        # 3 * [B, L, (HN + EN), HS]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, num_heads, _ = query_layer.shape

        # [B, L, (HN + EN), HS] -> [B, (HN + EN), L, HS] -> [B * (HN + EN), L, HS]
        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * num_heads, q_length, self.head_dim
        )
        # [B, L, (HN + EN), HS] -> [B, (HN + EN), HS, L] -> [B * (HN + EN), HS, L]
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(
            batch_size * num_heads, self.head_dim, q_length
        )
        # [B, L, (HN + EN), HS] -> [B, (HN + EN), L, HS] -> [B * (HN + EN), L, HS]
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * num_heads, q_length, self.head_dim
        )
        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [B * (HN + EN), HS, kv_length]
            #  - value: [B * (HN + EN), kv_length, HS]
            key_layer = torch.cat((past_key, key_layer), dim=2)
            value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        # [B * (HN + EN), L, HS] * [B * (HN + EN), HS, kv_length] -> [B * (HN + EN), L, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # [B * (HN + EN), L, kv_length] -> [B, (HN + EN), L, kv_length]
        attention_scores = matmul_result.view(
            batch_size, num_heads, q_length, kv_length
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [B, (HN + EN), L, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(
            attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min
        )
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            input_dtype
        )

        # [B, (HN + EN), L, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # [B, (HN + EN), L, kv_length] -> [B * (HN + EN), L, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * num_heads, q_length, kv_length
        )

        # [B * (HN + EN), L, kv_length] * [B * (HN + EN), kv_length, HS] -> [B * (HN + EN), L, HS]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # [B * (HN + EN), L, HS] -> [B, L, (HN + EN) * HS]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(
            output_tensor, residual, self.hidden_dropout, self.training
        )

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


def domain_enhance_att(
    model: MT, domain_att_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify BLOOM model to apply self attention domain enhancement.

    Args:
        model (Type[BloomPreTrainedModel]): Original BLOOM model class.
        domain_att_enhance (Optional[Dict[str, int]]): Domain enhancements.
            If None is provided, will read from existing configs.

    Returns:
        Type[BloomPreTrainedModel]: Patched model class
    """
    attention_module: str = (
        "h.*.self_attention"
        if issubclass(model, BloomModel)
        else "transformer.h.*.self_attention"
    )

    model = cast(MT, model)

    origin_init = model.__init__

    def patched_init(
        self: BloomPreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        # patch config if new enhancement provided
        if domain_att_enhance is not None:
            config.domain_att_enhance = domain_att_enhance

        config_with_enhance = cast(Config, config)
        # if domain enhance, replace modules
        if config_with_enhance.domain_att_enhance:
            nested_replace_module(
                self, attention_module, lambda _: BloomAttention(config_with_enhance)
            )

        self.post_init()

    model = type(
        f"{model.__name__}_EnhanceAtt", (model,), {"__init__": patched_init}
    )  # type: ignore

    return model
