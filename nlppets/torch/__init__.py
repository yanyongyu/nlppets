try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch module not installed. Please install it first."
    ) from None

from .combine_model import combine_model as combine_model
from .concat_linear import concat_linear as concat_linear
from .freeze_tensor import freeze_tensor as freeze_tensor
from .replace_module import replace_module as replace_module
from .freeze_tensor import unfreeze_tensor as unfreeze_tensor
from .freeze_tensor import nested_freeze_tensor as nested_freeze_tensor
from .count_cuda_devices import count_cuda_devices as count_cuda_devices
from .replace_module import nested_replace_module as nested_replace_module
from .freeze_tensor import nested_unfreeze_tensor as nested_unfreeze_tensor
