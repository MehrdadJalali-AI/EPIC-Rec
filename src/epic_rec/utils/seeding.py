"""Deterministic random number generation.

All randomness in EPIC-Rec goes through :func:`rng_for`, which derives an independent generator from a base seed
and a tuple of integers or strings (for example ``(seed, "bpr", user)``). Results therefore do not depend on the
order in which parallel workers process jobs.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's and NumPy's global generators (only used by third-party code; EPIC-Rec uses rng_for)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)


def rng_for(*key: object) -> np.random.Generator:
    """Return a generator whose stream depends only on ``key``."""
    digest = hashlib.sha256(repr(key).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))
