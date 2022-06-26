try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch module not installed. Please install it first."
    ) from None


from .combine_model import combine_model as combine_model
from .replace_module import replace_module as replace_module
from .replace_module import nested_replace_module as nested_replace_module
