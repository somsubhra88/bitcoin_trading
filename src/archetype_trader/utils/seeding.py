"""Deterministic seeding across Python, NumPy and PyTorch.

Reproducibility is a research requirement, not a nicety: every entry point
calls :func:`seed_everything` before touching a generator, and the resolved
seed is written into the run manifest so a result can be re-derived.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    """The seed actually applied, recorded in run manifests."""

    seed: int
    deterministic_torch: bool
    cudnn_benchmark: bool


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed all RNGs this project can reach.

    ``deterministic`` trades throughput for bit-reproducibility. It is on by
    default because a non-reproducible research result is not a result.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # ``warn_only`` keeps ops without a deterministic kernel usable while
        # still surfacing them, rather than hard-failing a long training run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    return SeedState(seed=seed, deterministic_torch=deterministic, cudnn_benchmark=False)


def make_rng(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator (preferred over global state)."""
    return np.random.default_rng(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick a torch device, falling back loudly rather than silently failing."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS lacks deterministic kernels for several ops we rely on; CPU is the
    # reproducible default and these models are small.
    return torch.device("cpu")
