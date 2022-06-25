try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch module not installed. Please install it first."
    ) from None


from .combine_model import combine_model as combine_model
