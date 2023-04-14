from contextvars import ContextVar
from typing import Tuple, Union, TypeVar, Callable, Optional

import torch.nn as nn

M = TypeVar("M", bound=nn.Module)
LOC: ContextVar[Tuple[str, ...]] = ContextVar("LOC")


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
    module: M,
    child_name: str,
    new_child: Union[
        None, nn.Module, Callable[[Tuple[str, ...], nn.Module], Optional[nn.Module]]
    ],
) -> M:
    """Replace a nested child module in a module.

    Child module name is a dot-separated string of module names.
    Use `*` to replace all child modules (like items of `nn.ModuleList`).

    Args:
        module (nn.Module): The module to replace the child in.
        child_name (str): The name of the child module to replace.
        new_child (Union[nn.Module, Callable[[Tuple[str, ...]], nn.Module]]):
            The new child module or a new module factory with module loc tuple as arg.

    Returns:
        nn.Module: The module with the child replaced.
    """
    current_location = LOC.get(tuple())
    nested_names = child_name.split(".")
    if len(nested_names) == 1:
        if new_child is not None and not isinstance(new_child, nn.Module):
            new_child = new_child(
                current_location + (nested_names[0],), getattr(module, nested_names[0])
            )
        if new_child is not None:
            replace_module(module, nested_names[0], new_child)
    elif nested_names[0] == "*":
        for name, child in module.named_children():
            token = LOC.set(current_location + (name,))
            try:
                nested_replace_module(child, ".".join(nested_names[1:]), new_child)
            finally:
                LOC.reset(token)
    else:
        child = getattr(module, nested_names[0])
        token = LOC.set(current_location + (nested_names[0],))
        try:
            nested_replace_module(child, ".".join(nested_names[1:]), new_child)
        finally:
            LOC.reset(token)

    return module
