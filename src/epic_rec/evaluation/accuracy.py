"""Top-K accuracy (Recall@K, NDCG@K) with the solver's ranking rule.

Ranking: score descending, ties by item id ascending. Training items are excluded from the ranking; when
evaluating on the test split, validation items are excluded as well (they are known positives that are not
test targets). Recall@K = hits / min(|test items|, K).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.data.dataset import InteractionDataset


def rank_topk(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[1]
    ids = np.broadcast_to(np.arange(n), scores.shape)
    order = np.lexsort((ids, -scores), axis=1)
    return order[:, :k]


def evaluate(model: Any, ds: InteractionDataset, split: str = "test", ks: tuple[int, ...] = (10, 20),
             batch: int = 512, users: np.ndarray | None = None) -> dict[str, float]:
    truth = ds.by_user(split)
    exclude = [ds.by_user("train")] + ([ds.by_user("val")] if split == "test" else [])
    us = np.array(sorted(truth)) if users is None else np.asarray(users)
    kmax = max(ks)
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))
    rec = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    for a in range(0, len(us), batch):
        ub = us[a : a + batch]
        sc = np.asarray(model.score_batch(ub), dtype=np.float64)
        for r, u in enumerate(ub):
            for ex in exclude:
                if u in ex:
                    sc[r, ex[u]] = -np.inf
        top = rank_topk(sc, kmax)
        for r, u in enumerate(ub):
            t = set(truth[u].tolist())
            hits = np.array([x in t for x in top[r]])
            for k in ks:
                h = hits[:k]
                rec[k].append(h.sum() / min(len(t), k))
                ndcg[k].append((h * disc[:k]).sum() / disc[: min(len(t), k)].sum())
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(rec[k]))
        out[f"ndcg@{k}"] = float(np.mean(ndcg[k]))
    out["users"] = int(len(us))
    return out
