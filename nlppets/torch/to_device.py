from typing import Mapping, TypeVar, Optional, Sequence

import torch
from torch import nn

T = TypeVar("T")


def to_device(
    obj: T,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> T:
    """Ensure the obj to specific device.

    Args:
        obj (T): The object to be converted to device.
        device (Optional[torch.device], optional): Specific device. Defaults to None.
        dtype (Optional[torch.dtype], optional): Optional change the dtype of obj. Defaults to None.

    Returns:
        T: The converted object.
    """
    if isinstance(obj, (torch.Tensor, nn.Module)):
        return obj.to(device, dtype)
    elif isinstance(obj, Mapping):
        return obj.__class__(
            (key, to_device(value, device, dtype)) for key, value in obj.items()  # type: ignore
        )
    elif isinstance(obj, Sequence):
        return obj.__class__(to_device(value, device, dtype) for value in obj)  # type: ignore

    return obj
