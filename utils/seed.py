"""Reproducibility: unified seeding of Python, NumPy, PyTorch (CPU+CUDA)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Seed every RNG used by the project.

    deterministic=True sets cuDNN to deterministic mode. This can slow
    training; all thesis experiments default to True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    """Worker init function for DataLoader (use with worker_init_fn)."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
