import torch.nn as nn


def get_param_numel(param: nn.Parameter, deepspeed: bool = False) -> int:
    return param.ds_numel if deepspeed else param.numel()  # type: ignore


def count_parameters(module: nn.Module, deepspeed: bool = False) -> int:
    return sum(
        max(0, get_param_numel(param, deepspeed=deepspeed))
        for _, param in module.named_parameters(recurse=True)
    )


def count_trainable_parameters(module: nn.Module, deepspeed: bool = False) -> int:
    return sum(
        max(0, get_param_numel(param, deepspeed=deepspeed))
        for _, param in module.named_parameters(recurse=True)
        if param.requires_grad
    )
