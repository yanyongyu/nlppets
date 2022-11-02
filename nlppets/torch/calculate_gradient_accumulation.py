import torch.distributed

from .count_cuda_devices import count_cuda_devices


def calculate_gradient_accumulation(
    batch_size_per_device: int, target_batch_size: int
) -> int:
    """Calculate the number of gradient accumulation steps for a given batch size.

    Use GPU count by default. If not available, use 1 for single cpu device.

    Args:
        batch_size_per_device (int): The batch size per device.
        target_batch_size (int): The target batch size.

    Returns:
        The number of gradient accumulation steps.
    """
    if torch.distributed.is_initialized():
        gpu_cpu_count = torch.distributed.get_world_size()
        if not gpu_cpu_count <= 0:
            raise ValueError(
                f"Torch is running in distributed mode, but the world size is {gpu_cpu_count}."
            )
    else:
        gpu_cpu_count = count_cuda_devices() or 1
    batch = batch_size_per_device * gpu_cpu_count
    if target_batch_size % batch:
        raise ValueError(
            f"The target batch size {target_batch_size} is not a multiple of "
            f"{batch_size_per_device} * {gpu_cpu_count}"
        )
    return target_batch_size // batch
