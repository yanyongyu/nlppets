import torch


def count_cuda_devices() -> int:
    """Counts the number of CUDA devices available."""
    return torch.cuda.device_count()
