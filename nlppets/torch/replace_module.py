from typing import Union, TypeVar, Callable

import torch.nn as nn

M = TypeVar("M", bound=nn.Module)


def replace_module(module: M, child_name: str, new_child: nn.Module) -> M:
    """Replace a child module in a module.

    Args:
        module (nn.Module): The module to replace the child in.
        child_name (str): The name of the child module to replace.
        new_child (nn.Module): The new child module.

    Raises:
        ValueError: If the child module name is invalid.
        AttributeError: If the child module does not exist.

    Returns:
        nn.Module: The module with the child replaced.
    """
    if not child_name:
        raise ValueError(f"Invalid child module name: {child_name}")

    # ensure module exists
    module.get_submodule(child_name)
    # replace module
    setattr(module, child_name, new_child)
    return module


def nested_replace_module(
    module: M, child_name: str, new_child: Union[nn.Module, Callable[[], nn.Module]]
) -> M:
    """Replace a nested child module in a module.

    Child module name is a dot-separated string of module names.
    Use `*` to replace all child module (like items of `nn.ModuleList`).

    Args:
        module (nn.Module): The module to replace the child in.
        child_name (str): The name of the child module to replace.
        new_child (Union[nn.Module, Callable[[], nn.Module]]):
            The new child module or a new module factory.

    Returns:
        nn.Module: The module with the child replaced.
    """
    nested_names = child_name.split(".")
    if len(nested_names) == 1:
        if not isinstance(new_child, nn.Module):
            new_child = new_child()
        replace_module(module, nested_names[0], new_child)
    elif nested_names[0] == "*":
        for _, child in module.named_children():
            nested_replace_module(child, ".".join(nested_names[1:]), new_child)
    else:
        child = getattr(module, nested_names[0])
        nested_replace_module(child, ".".join(nested_names[1:]), new_child)
    return module
