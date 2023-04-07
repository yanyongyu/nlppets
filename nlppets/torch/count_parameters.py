import torch.nn as nn


def count_parameters(module: nn.Module):
    return sum(
        max(0, param.numel()) for _, param in module.named_parameters(recurse=True)
    )


def count_trainable_parameters(module: nn.Module):
    return sum(
        max(0, param.numel())
        for _, param in module.named_parameters(recurse=True)
        if param.requires_grad
    )
