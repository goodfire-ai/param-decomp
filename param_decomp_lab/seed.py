import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Seed `random`, NumPy, and PyTorch's global RNGs.

    No-op when `seed` is ``None``.

    Args:
        seed: Integer seed shared across all three libraries.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
