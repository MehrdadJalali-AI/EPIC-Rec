"""Small synthetic dataset for unit tests, continuous integration and the demo.

Users belong to latent taste groups; items carry a "genre" entity per group and "creator" entities shared by a
few items, so the knowledge graph creates paths with different strengths. Everything is generated from the seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from epic_rec.data.dataset import InteractionDataset, split_per_user
from epic_rec.utils.seeding import rng_for


def make_synthetic(
    n_users: int = 60,
    n_items: int = 80,
    n_groups: int = 4,
    n_creators: int = 25,
    mean_history: int = 14,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
) -> InteractionDataset:
    rng = rng_for(seed, "synthetic")
    item_group = rng.integers(0, n_groups, n_items)
    user_group = rng.integers(0, n_groups, n_users)
    pop = rng.pareto(1.5, n_items) + 1
    pairs = []
    for u in range(n_users):
        n = int(np.clip(rng.poisson(mean_history), 10, n_items // 2))
        w = pop * np.where(item_group == user_group[u], 6.0, 1.0)
        items = rng.choice(n_items, n, replace=False, p=w / w.sum())
        pairs += [(u, int(i)) for i in items]
    pairs = np.array(sorted(pairs), dtype=np.int32)
    links = [(i, "genre", int(g)) for i, g in enumerate(item_group)]
    second = rng.random(n_items) < 0.3
    links += [(i, "genre", int((item_group[i] + 1) % n_groups)) for i in np.flatnonzero(second)]
    creator = rng.integers(0, n_creators, n_items)
    links += [(i, "creator", n_groups + int(c)) for i, c in enumerate(creator)]
    ie = pd.DataFrame(links, columns=["item", "relation", "entity"]).drop_duplicates()
    deg = ie.groupby(["relation", "entity"]).item.transform("nunique")
    ie = ie[deg >= 2].sort_values(["relation", "entity", "item"]).reset_index(drop=True)
    train, val, test = split_per_user(pairs, n_users, val_frac, test_frac, seed)
    meta = dict(source=dict(description="synthetic dataset (epic_rec.data.synthetic)", linkage="synthetic"),
                config=dict(n_users=n_users, n_items=n_items, n_groups=n_groups, n_creators=n_creators,
                            mean_history=mean_history, seed=seed))
    ds = InteractionDataset("synthetic", n_users, n_items, train, val, test, ie, np.arange(n_users),
                            np.arange(n_items), meta)
    ds.check_no_leakage()
    return ds
