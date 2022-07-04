try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch module not installed. Please install it first."
    ) from None


from .combine_model import combine_model as combine_model
from .freeze_tensor import freeze_tensor as freeze_tensor
from .replace_module import replace_module as replace_module
from .freeze_tensor import unfreeze_tensor as unfreeze_tensor
from .freeze_tensor import nested_freeze_tensor as nested_freeze_tensor
from .replace_module import nested_replace_module as nested_replace_module
