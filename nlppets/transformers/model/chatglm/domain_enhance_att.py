import math
import inspect
from functools import wraps
from contextvars import ContextVar
from typing import Any, Dict, Type, Tuple, TypeVar, Callable, Optional, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from nlppets.general import MonkeyPatch
from nlppets.torch import concat_linear

MT = TypeVar("MT", bound=Type[PreTrainedModel])


model_config: ContextVar["Config"] = ContextVar("model_config")
model_attention: ContextVar[Type[nn.Module]] = ContextVar("model_attention")


class Config(Protocol):
    hidden_size: int
    num_attention_heads: int
    domain_att_enhance: Dict[str, int]
    """domain pre-training enhancements."""


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


@torch.jit.script
def apply_rotary_pos_emb_index(
    q: torch.Tensor,  # [L, B, (HN + EN), HS]
    k: torch.Tensor,  # [L, B, (HN + EN), HS]
    cos: torch.Tensor,  # [L, 1, HS]
    sin: torch.Tensor,  # [L, 1, HS]
    position_id: torch.Tensor,  # [L, B]
) -> Tuple[torch.Tensor, torch.Tensor]:  # 2 * [L, B, (HN + EN), HS]
    # 2 * [L, B, 1, HS]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(
        position_id, sin.squeeze(1)
    ).unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class ChatGLMAttention(nn.Module):
    layer_id: int
    hidden_size: int
    hidden_size_per_partition: int
    hidden_size_per_attention_head: int
    inner_hidden_size: int
    num_attention_heads: int
    num_attention_heads_per_partition: int
    position_encoding_2d: bool
    rotary_emb: Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    scale_mask_softmax: Any
    query_key_value: torch.nn.Linear
    dense: torch.nn.Linear

    def __init__(self, *args, **kwargs):
        model_attention.get().__init__(self, *args, **kwargs)

        self.enhancements = model_config.get().domain_att_enhance
        self.additional_heads = sum(self.enhancements.values())
        for name, size in model_config.get().domain_att_enhance.items():
            setattr(
                self,
                f"{name}",
                nn.Linear(
                    model_config.get().hidden_size,
                    self.hidden_size_per_attention_head * size * 3,
                ),
            )
            setattr(
                self,
                f"{name}_output",
                nn.Linear(
                    self.hidden_size_per_attention_head * size,
                    model_config.get().hidden_size,
                ),
            )

    def split_tensor_along_last_dim(
        self,
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
    ):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_id: int,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # [L, B, H] -> [L, B, (HN + EN) * 3 * HS]
        mixed_raw_layer = concat_linear(
            self.query_key_value,
            *(getattr(self, name) for name in self.enhancements.keys()),
        )(hidden_states)

        # [L, B, (HN + EN) * 3 * HS] -> [L, B, (HN + EN), 3 * HS]
        new_tensor_shape = mixed_raw_layer.size()[:-1] + (
            self.num_attention_heads_per_partition + self.additional_heads,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_raw_layer = mixed_raw_layer.view(*new_tensor_shape)

        # 3 * [L, B, (HN + EN), HS]
        (query_layer, key_layer, value_layer) = self.split_tensor_along_last_dim(
            mixed_raw_layer, 3
        )

        if self.position_encoding_2d:
            # [L, B, (HN + EN), HS] -> 2 * [L, B, (HN + EN), HS // 2]
            q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
            # [L, B, (HN + EN), HS] -> 2 * [L, B, (HN + EN), HS // 2]
            k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
            cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
            position_ids, block_position_ids = (
                position_ids[:, 0, :].transpose(0, 1).contiguous(),
                position_ids[:, 1, :].transpose(0, 1).contiguous(),
            )
            q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
            q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
            # [L, B, (HN + EN), HS]
            query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
            key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
        else:
            position_ids = position_ids.transpose(0, 1)
            cos, sin = self.rotary_emb(value_layer, seq_len=position_ids.max() + 1)
            # [L, B, (HN + EN), HS]
            query_layer, key_layer = apply_rotary_pos_emb_index(
                query_layer, key_layer, cos, sin, position_ids
            )

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension:
            #  - key: [kv_length, B, (HN + EN), HS]
            #  - value: [kv_length, B, (HN + EN), HS]
            key_layer = torch.cat((past_key, key_layer), dim=0)
            value_layer = torch.cat((past_value, value_layer), dim=0)

        # [kv_length, B, (HN + EN), HS]
        seq_length = query_layer.shape[0]
        kv_length, batch_size, num_heads, hidden_size = key_layer.shape

        present = (key_layer, value_layer) if use_cache else None

        scaling_attention_score = True
        query_key_layer_scaling_coeff = float(layer_id + 1)
        if scaling_attention_score:
            query_layer = query_layer / (
                math.sqrt(hidden_size) * query_key_layer_scaling_coeff
            )

        # [L, B, (HN + EN), HS] -> [L, B * (HN + EN), HS]
        query_layer = query_layer.view(
            seq_length,
            batch_size * num_heads,
            hidden_size,
        )
        # [kv_length, B, (HN + EN), HS] -> [kv_length, B * (HN + EN), HS]
        key_layer = key_layer.view(kv_length, batch_size * num_heads, hidden_size)

        matmul_result = torch.zeros(
            1,
            1,
            1,
            dtype=query_layer.dtype,
            device=query_layer.device,
        )
        # [B * (HN + EN), L, HS] * [B * (HN + EN), HS, kv_length] -> [B * (HN + EN), L, kv_length]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [B * (HN + EN), L, HS]
            key_layer.transpose(0, 1).transpose(1, 2),  # [B * (HN + EN), HS, kv_length]
            beta=0.0,
            alpha=1.0,
        )

        # [B * (HN + EN), L, kv_length] -> [B, (HN + EN), L, kv_length]
        attention_scores = matmul_result.view(
            batch_size, num_heads, seq_length, kv_length
        )

        if self.scale_mask_softmax:
            self.scale_mask_softmax.scale = query_key_layer_scaling_coeff
            attention_probs: torch.Tensor = self.scale_mask_softmax(
                attention_scores, attention_mask.contiguous()
            )
        else:
            if not (attention_mask == 0).all():
                # if auto-regressive, skip
                attention_scores.masked_fill_(attention_mask, -10000.0)
            dtype = attention_scores.dtype
            attention_scores = attention_scores.float()
            attention_scores = attention_scores * query_key_layer_scaling_coeff

            attention_probs = F.softmax(attention_scores, dim=-1)

            attention_probs = attention_probs.type(dtype)

        # context layer shape: [b, np, sq, hn]
        # output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))

        # [kv_length, B, (HN + EN), HS] -> [kv_length, B * (HN + EN), HS]
        value_layer = value_layer.view(kv_length, batch_size * num_heads, hidden_size)

        # [B, (HN + EN), L, kv_length] -> [B * (HN + EN), L, kv_length]
        attention_probs = attention_probs.view(
            batch_size * num_heads, seq_length, kv_length
        )

        # [B * (HN + EN), L, kv_length] * [B * (HN + EN), kv_length, HS] -> [B * (HN + EN), L, HS]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # [B * (HN + EN), L, HS] -> [B, (HN + EN), L, HS]
        context_layer = context_layer.view(
            batch_size, num_heads, seq_length, hidden_size
        )

        # [B, (HN + EN), L, HS] -> [L, B, (HN + EN), HS]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [L, B, (HN + EN), HS] -> [L, B, (HN + EN) * HS]
        new_context_layer_shape = context_layer.size()[:-2] + (
            num_heads * self.hidden_size_per_attention_head,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        output = concat_linear(
            self.dense,
            *(getattr(self, f"{name}_output") for name in self.enhancements.keys()),
        )(context_layer)

        outputs = (output, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs


def domain_enhance_att(
    model: MT, domain_att_enhance: Optional[Dict[str, int]] = None
) -> MT:
    """Modify ChatGLM model to apply self attention domain enhancement.

    Args:
        model (Type[ChatGLMPreTrainedModel]): Original ChatGLM model class.
        domain_att_enhance (Optional[Dict[str, int]]): Domain enhancements.
            If None is provided, will read from existing configs.

    Returns:
        Type[ChatGLMPreTrainedModel]: Patched model class
    """

    model = cast(MT, model)

    origin_init = model.__init__
    chatglm_module = inspect.getmodule(model)
    origin_attention = getattr(chatglm_module, "SelfAttention")

    def patched_init(self: PreTrainedModel, config: PretrainedConfig, *args, **kwargs):
        # patch config if new enhancement provided
        if domain_att_enhance is not None:
            config.domain_att_enhance = domain_att_enhance

        config_with_enhance = cast(Config, config)

        with MonkeyPatch.context() as m:
            m.setattr(chatglm_module, "SelfAttention", ChatGLMAttention)
            t1 = model_attention.set(origin_attention)
            t2 = model_config.set(config_with_enhance)

            try:
                origin_init(self, config, *args, **kwargs)
            finally:
                model_attention.reset(t1)
                model_config.reset(t2)

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
        type(f"{model.__name__}_EnhanceAtt", (model,), {"__init__": patched_init})
    )  # type: ignore
