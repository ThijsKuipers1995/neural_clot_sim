import torch
import numpy as np


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
