import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Set the random seed for `random`, NumPy, and PyTorch."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
