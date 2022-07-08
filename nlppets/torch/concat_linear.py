from functools import partial
from typing import Literal, Callable

import torch
import torch.nn as nn
from torch.nn.functional import linear


def concat_linear(*layers: nn.Linear) -> Callable[[torch.Tensor], torch.Tensor]:
    # check layer bias
    use_bias: bool
    if all(layer.bias is not None for layer in layers):
        use_bias = True
    elif all(layer.bias is None for layer in layers):
        use_bias = False
    else:
        raise ValueError("Bias must be either all None or all not None")

    concat_type: Literal["in", "out"]
    if len({layer.in_features for layer in layers}) == 1:
        concat_type = "in"
    elif len({layer.out_features for layer in layers}) == 1:
        concat_type = "out"
    else:
        raise ValueError("Layers must have either same in_features or out_features")

    if concat_type == "in":
        # concat weight and bias in the output feature dim
        concat_weight = torch.concat(tuple(layer.weight for layer in layers), dim=0)
        concat_bias = (
            torch.cat(tuple(layer.bias for layer in layers), dim=0)
            if use_bias
            else None
        )
    else:
        # concat weight and bias in the input featue dim
        concat_weight = torch.concat(tuple(layer.weight for layer in layers), dim=1)
        concat_bias = (
            torch.stack(tuple(layer.bias for layer in layers), dim=0).sum(dim=0)
            if use_bias
            else None
        )
    return partial(linear, weight=concat_weight, bias=concat_bias)
