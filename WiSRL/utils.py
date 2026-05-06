import random
import numpy as np
from typing import Tuple

import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Ensure each DataLoader worker uses a deterministic seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute NT-Xent loss for a batch of positive pairs.

    - Positive: z1[i] with z2[i] (same image, different views)
    - Negative: all other samples in the batch
    来自于SimCLR论文的损失函数实现，适用于对比学习中的正负样本构造和损失计算。
    """

    assert z1.size() == z2.size(), "Input tensors must have the same shape"

    batch_size = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # (2N, D) -> similarity matrix (2N, 2N)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * batch_size, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # Positive pairs are (i -> i+N) and (i+N -> i)
    labels = torch.arange(2 * batch_size, device=sim.device)
    labels[:batch_size] = labels[:batch_size] + batch_size
    labels[batch_size:] = labels[batch_size:] - batch_size
    loss = F.cross_entropy(sim, labels)
    return loss
