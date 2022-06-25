from typing import Mapping, OrderedDict

import torch


def combine_model(
    old: Mapping[str, torch.Tensor], new: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Combine two models' state dict.

    Second takes higher priority.

    Args:
        old (Mapping[str, torch.Tensor]): The old model state dict to combine.
        new (Mapping[str, torch.Tensor]): The new model state dict to combine.

    Returns:
        OrderedDict[str, torch.Tensor]: The combined model state dict.
    """
    # torch.nn.Module.load_state_dict accepts an OrderedDict instead of simple dict.
    return OrderedDict([*old.items(), *new.items()])
