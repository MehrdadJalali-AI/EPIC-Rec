"""Processed dataset container: interactions, deterministic splits and the item-entity table."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from epic_rec.utils.io import atomic_write_json, sha256_file
from epic_rec.utils.seeding import rng_for


@dataclass
class InteractionDataset:
    """Implicit-feedback interactions with a KG item-entity table.

    ``train``, ``val`` and ``test`` are integer arrays of shape (n, 2) holding (user, item) pairs with internal ids
    0..n_users-1 and 0..n_items-1. ``item_entity`` has columns ``item, relation, entity`` (internal item ids,
    canonical relation names, entity ids). ``meta`` records source, filters, seed and statistics.
    """

    name: str
    n_users: int
    n_items: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    item_entity: pd.DataFrame
    user_ids: np.ndarray
    item_ids: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ matrices
    def matrix(self, split: str = "train") -> sp.csr_matrix:
        pairs = getattr(self, split)
        data = np.ones(len(pairs), np.float32)
        return sp.csr_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(self.n_users, self.n_items))

    def by_user(self, split: str) -> dict[int, np.ndarray]:
        pairs = getattr(self, split)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        p = pairs[order]
        cuts = np.flatnonzero(np.diff(p[:, 0])) + 1
        return {int(g[0, 0]): g[:, 1] for g in np.split(p, cuts) if len(g)}

    def stats(self) -> dict[str, Any]:
        rels = self.item_entity.groupby("relation").size().sort_values(ascending=False)
        hist = np.bincount(self.train[:, 0], minlength=self.n_users)
        return dict(
            users=int(self.n_users),
            items=int(self.n_items),
            train=int(len(self.train)),
            val=int(len(self.val)),
            test=int(len(self.test)),
            density=float((len(self.train) + len(self.val) + len(self.test)) / (self.n_users * self.n_items)),
            train_history_median=float(np.median(hist)),
            kg_links=int(len(self.item_entity)),
            kg_entities=int(self.item_entity.entity.nunique()) if len(self.item_entity) else 0,
            kg_relations={str(k): int(v) for k, v in rels.items()},
            items_with_kg=int(self.item_entity.item.nunique()) if len(self.item_entity) else 0,
        )

    # ------------------------------------------------------------------ checks
    def check_no_leakage(self) -> None:
        """Raise if a (user, item) pair occurs in more than one split or twice in one split."""
        keys = {}
        for split in ("train", "val", "test"):
            p = getattr(self, split)
            k = p[:, 0].astype(np.int64) * self.n_items + p[:, 1]
            if len(np.unique(k)) != len(k):
                raise AssertionError(f"duplicate pairs inside {split}")
            keys[split] = set(k.tolist())
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if keys[a] & keys[b]:
                raise AssertionError(f"{len(keys[a] & keys[b])} pairs shared by {a} and {b}")

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            d / "interactions.npz",
            train=self.train,
            val=self.val,
            test=self.test,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            n=np.array([self.n_users, self.n_items]),
        )
        self.item_entity.to_csv(d / "item_entity.csv.gz", index=False)
        meta = dict(self.meta)
        meta["name"] = self.name
        meta["stats"] = self.stats()
        meta["files"] = {f: sha256_file(d / f) for f in ("interactions.npz", "item_entity.csv.gz")}
        atomic_write_json(d / "manifest.json", meta)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> InteractionDataset:
        import json

        d = Path(directory)
        z = np.load(d / "interactions.npz", allow_pickle=False)
        meta = json.loads((d / "manifest.json").read_text())
        ie = pd.read_csv(d / "item_entity.csv.gz", dtype={"relation": str})
        n_users, n_items = (int(x) for x in z["n"])
        return cls(
            name=meta.get("name", d.name),
            n_users=n_users,
            n_items=n_items,
            train=z["train"],
            val=z["val"],
            test=z["test"],
            item_entity=ie,
            user_ids=z["user_ids"],
            item_ids=z["item_ids"],
            meta=meta,
        )


def split_per_user(
    pairs: np.ndarray, n_users: int, val_frac: float, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random per-user split. Each user's items are sorted, then shuffled with a generator derived from
    (seed, user), so the split does not depend on input order. Every user keeps at least one training item."""
    train, val, test = [], [], []
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    p = pairs[order]
    cuts = np.flatnonzero(np.diff(p[:, 0])) + 1
    for g in np.split(p, cuts):
        if not len(g):
            continue
        u = int(g[0, 0])
        items = g[:, 1].copy()
        rng_for(seed, "split", u).shuffle(items)
        n = len(items)
        n_te = int(round(test_frac * n))
        n_va = int(round(val_frac * n))
        if n - n_te - n_va < 1:
            n_te = max(0, min(n_te, n - 1))
            n_va = max(0, n - 1 - n_te) if n - n_te - n_va < 1 else n_va
        test.append(np.column_stack([np.full(n_te, u), items[:n_te]]))
        val.append(np.column_stack([np.full(n_va, u), items[n_te : n_te + n_va]]))
        train.append(np.column_stack([np.full(n - n_te - n_va, u), items[n_te + n_va :]]))

    def cat(x: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(x).astype(np.int32) if x else np.zeros((0, 2), np.int32)

    return cat(train), cat(val), cat(test)
