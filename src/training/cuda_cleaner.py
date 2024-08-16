import torch


def clean_cuda_memory_by_threshold(memory_threshold_gb: float) -> bool:
    """
    return True if we need to clean GPU memory
    """
    # GPU Memory profiling
    allocated_memory = torch.cuda.memory_allocated() / (1024**3)  # in GB
    reserved_memory = torch.cuda.memory_reserved() / (1024**3)  # in GB
    # Optionally clear cache if memory usage is high
    if allocated_memory > memory_threshold_gb:
        return True

    return False
