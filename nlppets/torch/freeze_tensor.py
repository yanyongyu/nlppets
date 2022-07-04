from typing import Set, Dict, TypeVar, Iterable, Optional

import torch.nn as nn

M = TypeVar("M", bound=nn.Module)


def freeze_tensor(
    module: M,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> M:
    """Freeze specific tensors in a module.

    Args:
        module (nn.Module): Module to freeze tensors in.
        include (Optional[Iterable[str]], optional): Tensors to freeze. Defaults to all tensors.
        exclude (Optional[Iterable[str]], optional): Tensors to exclude. Defaults to None.

    Returns:
        nn.Module: Module with frozen tensors.
    """
    include = set() if include is None else set(include)
    exclude = set() if exclude is None else set(exclude)
    for name, param in module.named_parameters(recurse=False):
        if (not include or name in include) and name not in exclude:
            param.requires_grad = False

    return module


def _set_to_prefix_dict(input: Set[str]) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for name in input:
        prefix, *remain = name.split(".")
        result.setdefault(prefix, set()).add(".".join(remain))
    return result


def nested_freeze_tensor(
    module: M,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> M:
    """Nested freeze specific tensors in a module.

    Tensor name is a dot-separated string of module names.
    Use `*` to nested freeze all children.

    Args:
        module (nn.Module): Module to freeze tensors in.
        include (Optional[Iterable[str]], optional): Tensors to freeze. Defaults to all tensors.
        exclude (Optional[Iterable[str]], optional): Tensors to exclude. Defaults to None.

    Returns:
        nn.Module: Module with frozen tensors.
    """
    include = {"*"} if include is None else set(include)
    exclude = set() if exclude is None else set(exclude)

    # return if include nothing
    if not include:
        return module

    # return if exclude all
    if "*" in exclude:
        return module

    include_children = _set_to_prefix_dict(include)
    exclude_children = _set_to_prefix_dict(exclude)

    # freeze all children
    if "*" in include:
        # first freeze current module's tensors
        freeze_tensor(module, exclude=exclude_children.keys())
        for name, child in module.named_children():
            nested_freeze_tensor(
                child, exclude=exclude_children.get(name, exclude_children.get("*"))
            )

    # include something
    # first freeze current module's tensors
    freeze_tensor(module, include_children.keys(), exclude_children.keys())
    for name, child in module.named_children():
        if name in include_children:
            nested_freeze_tensor(
                child,
                include_children[name],
                exclude_children.get(name, exclude_children.get("*")),
            )
    return module


def unfreeze_tensor(
    module: M,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> M:
    """Unfreeze specific tensors in a module.

    Args:
        module (nn.Module): Module to unfreeze tensors in.
        include (Optional[Iterable[str]], optional): Tensors to unfreeze. Defaults to all tensors.
        exclude (Optional[Iterable[str]], optional): Tensors to exclude. Defaults to None.

    Returns:
        nn.Module: Module with unfreezed tensors.
    """
    include = set() if include is None else set(include)
    exclude = set() if exclude is None else set(exclude)
    for name, param in module.named_parameters(recurse=False):
        if (not include or name in include) and name not in exclude:
            param.requires_grad = True

    return module


def nested_unfreeze_tensor(
    module: M,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> M:
    """Nested unfreeze specific tensors in a module.

    Tensor name is a dot-separated string of module names.
    Use `*` to nested unfreeze all children.

    Args:
        module (nn.Module): Module to unfreeze tensors in.
        include (Optional[Iterable[str]], optional): Tensors to unfreeze. Defaults to all tensors.
        exclude (Optional[Iterable[str]], optional): Tensors to exclude. Defaults to None.

    Returns:
        nn.Module: Module with frozen tensors.
    """
    include = {"*"} if include is None else set(include)
    exclude = set() if exclude is None else set(exclude)

    # return if include nothing
    if not include:
        return module

    # return if exclude all
    if "*" in exclude:
        return module

    include_children = _set_to_prefix_dict(include)
    exclude_children = _set_to_prefix_dict(exclude)

    # unfreeze all children
    if "*" in include:
        # first unfreeze current module's tensors
        unfreeze_tensor(module, exclude=exclude_children.keys())
        for name, child in module.named_children():
            nested_unfreeze_tensor(
                child, exclude=exclude_children.get(name, exclude_children.get("*"))
            )

    # include something
    # first unfreeze current module's tensors
    unfreeze_tensor(module, include_children.keys(), exclude_children.keys())
    for name, child in module.named_children():
        if name in include_children:
            nested_unfreeze_tensor(
                child,
                include_children[name],
                exclude_children.get(name, exclude_children.get("*")),
            )
    return module
