import math
from typing import Any, List, Type, Tuple, Literal, TypeVar, Optional, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.models.bert import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertSelfOutput as BaseSelfOutput
from transformers.models.bert.modeling_bert import (
    BertSelfAttention as BaseSelfAttention,
)

from nlppets.torch import nested_replace_module

MT = TypeVar("MT", bound=Type[BertPreTrainedModel])


class Config(Protocol):
    hidden_size: int
    num_attention_heads: int
    domain_att_enhance: List[str]
    """domain pre-training enhancements."""


class BertSelfAttention(BaseSelfAttention):
    def __init__(self, config: "Config"):
        super(BertSelfAttention, self).__init__(config)

        # added
        self.enhancements = config.domain_att_enhance
        for name in config.domain_att_enhance:
            setattr(
                self,
                f"{name}_query",
                nn.Linear(config.hidden_size, self.attention_head_size),
            )
            setattr(
                self,
                f"{name}_key",
                nn.Linear(config.hidden_size, self.attention_head_size),
            )
            setattr(
                self,
                f"{name}_value",
                nn.Linear(config.hidden_size, self.attention_head_size),
            )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, H] -> [B, L, HN, HS]
        new_x_shape = x.size()[:-1] + (
            x.size()[-1]
            // self.attention_head_size,  # calculate head num automatically
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        # [B, L, HN, HS] -> [B, HN, L, HS]
        return x.permute(0, 2, 1, 3)

    def _patch_for_domain_enhance(
        self, hidden_states: torch.Tensor, type: Literal["query", "key", "value"]
    ) -> torch.Tensor:
        # [B, L, H] -> [B, L, H]
        tmp_tensor = getattr(self, type)(hidden_states)
        for name in self.enhancements:
            # [B, L, H] -> [B, L, H + E]
            tmp_tensor = torch.concat(
                (tmp_tensor, getattr(self, f"{name}_{type}")(hidden_states)),
                dim=-1,
            )
        return tmp_tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Any, ...]:
        # [B, L, H] -> [B, L, H]
        mixed_query_layer = self._patch_for_domain_enhance(hidden_states, "query")

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k, v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(encoder_hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(encoder_hidden_states, "value")
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "value")
            )
            # concat over the length dim
            key_layer = torch.concat((past_key_value[0], key_layer), dim=2)
            value_layer = torch.concat((past_key_value[1], value_layer), dim=2)
        else:
            key_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "key")
            )
            value_layer = self.transpose_for_scores(
                self._patch_for_domain_enhance(hidden_states, "value")
            )

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # [B, HN + E, L, HS], [B, HN + E, L, HS] -> [B, HN + E, L, L]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type in {"relative_key", "relative_key_query"}:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs: torch.Tensor = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # [B, HN + E, L, L], [B, HN + E, L, HS] -> [B, HN + E, L, HS]
        context_layer = torch.matmul(attention_probs, value_layer)

        # [B, HN + E, L, HS] -> [B, L, H + E * HS]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size + len(self.enhancements) * self.attention_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(BaseSelfOutput):
    def __init__(self, config: "Config"):
        super(BertSelfOutput, self).__init__(config)

        # added
        self.enhancements = config.domain_att_enhance
        self.hidden_size = config.hidden_size
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        for name in config.domain_att_enhance:
            setattr(self, name, nn.Linear(self.attention_head_size, config.hidden_size))

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        # patched for domain enhancement
        # [B, L, H + E * HS] -> [B, L, H], E * [B, L, HS]
        hidden_states, *enhancements = torch.split(
            hidden_states,
            [self.hidden_size] + len(self.enhancements) * [self.attention_head_size],
            dim=-1,
        )

        # [B, L, H] -> [B, L, H]
        hidden_states = self.dense(hidden_states)

        # patched for domain enhancements
        for name, states in zip(self.enhancements, enhancements):
            # [B, L, HS] -> [B, L, H]
            # add together
            hidden_states += getattr(self, name)(states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def domain_enhance_att(model: MT, domain_att_enhance: Optional[List[str]] = None) -> MT:
    """Modify BERT model to apply self attention domain enhancement.

    Args:
        model (Type[BertPreTrainedModel]): Original BERT model class.
        domain_att_enhance (Optional[List[str]]): Domain enhancements.
            If None is provided, will read from existing configs.

    Returns:
        Type[BertPreTrainedModel]: Patched model class
    """
    attention_module: str = (
        "encoder.layer.*.attention.self"
        if model is BertModel
        else "bert.encoder.layer.*.attention.self"
    )
    attention_output_module: str = (
        "encoder.layer.*.attention.output"
        if model is BertModel
        else "bert.encoder.layer.*.attention.output"
    )

    origin_init = model.__init__

    def patched_init(
        self: PreTrainedModel, config: PretrainedConfig, *inputs, **kwargs
    ):
        origin_init(self, config, *inputs, **kwargs)

        # patch config if new enhancement provided
        if domain_att_enhance is not None:
            config.domain_att_enhance = domain_att_enhance

        config_with_enhance = cast(Config, config)
        # if domain enhance, replace modules
        if config_with_enhance.domain_att_enhance:
            nested_replace_module(
                self, attention_module, lambda: BertSelfAttention(config_with_enhance)
            )
            nested_replace_module(
                self,
                attention_output_module,
                lambda: BertSelfOutput(config_with_enhance),
            )

        self.post_init()

    model.__init__ = patched_init

    return model
