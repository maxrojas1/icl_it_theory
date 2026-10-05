import torch


def log_mem(stage: str = ""
) -> None:
    """
    Prints allocated and reserved CUDA memory usage.

    Args:
        stage: Pipeline stage name or checkpoint description.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"   [MEMORY] {stage:<20} | Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def log_exp(**kwargs
) -> None:
    """
    Prints key-value pairs of experimental hyperparameters in a formatted block.

    Args:
        **kwargs: Experiment metadata and hyperparameters to print.
    """
    print("\n")
    for name, value in kwargs.items():
        print(f"   {name}: {value}")
    print("\n")
