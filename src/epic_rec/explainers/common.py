"""Shared pieces of the approximate explainers: a violation surrogate and interaction orders."""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting


def violation(inst: ExplanationInstance, M: Any, setting: str | Setting, eps: float = EPS) -> float:
    """Non-negative surrogate of how far M is from satisfying C1 (and C2): 0 iff the margin conditions hold."""
    st = get_setting(setting)
    s = inst.masked_scores(M)
    t = inst.t
    if not st.c2:
        elig = np.flatnonzero(inst.eligible)
        others = s[elig[elig != t]]
        kth = np.partition(others, -inst.K)[-inst.K]
        return float(max(0.0, s[t] - kth + eps))
    so = np.sort(s[inst.outsiders])[::-1]
    b1 = so[0] if len(so) else -np.inf
    b2 = so[1] if len(so) > 1 else -np.inf
    v = max(0.0, s[t] - b1 + eps)
    v += float(np.maximum(0.0, max(s[t], b2) - s[inst.retained] + eps).sum())
    return float(v)


def first_outside(inst: ExplanationInstance, s: np.ndarray) -> int:
    """Highest-ranked candidate outside the current top-K (rank K + 1 under the id tie-break)."""
    idx = np.flatnonzero(inst.eligible)
    order = idx[np.lexsort((idx, -s[idx]))]
    return int(order[inst.K])


def item_popularity(inst: ExplanationInstance) -> np.ndarray:
    """Training popularity of every item (ones when the instance has no model)."""
    if inst.model is not None:
        return np.asarray(inst.model.pop, dtype=float)
    return np.ones(len(inst.s))


def static_order(inst: ExplanationInstance, criterion: str, rng: np.random.Generator | None = None) -> list[int]:
    """Fixed removal orders over D_u(t) (item ids). These are in-house heuristics, not published methods.

    target_score : descending A[i, t], the interaction's contribution to the target's score
    list_aware   : descending A[i, t] - mean_j A[i, j] over retained items j
    popularity   : descending training popularity of the interaction's item (control)
    random       : uniform random permutation (control)
    """
    sc = np.asarray(inst.scope)
    rows = inst.rows(sc)
    if criterion == "target_score":
        key = inst.A[rows, inst.t]
    elif criterion == "list_aware":
        key = inst.A[rows, inst.t] - inst.A[rows][:, inst.retained].mean(1)
    elif criterion == "popularity":
        key = item_popularity(inst)[sc]
    elif criterion == "random":
        return [int(i) for i in (rng or np.random.default_rng(0)).permutation(sc)]
    else:
        raise ValueError(f"unknown static criterion {criterion}")
    return [int(i) for i in sc[np.argsort(-key, kind="stable")]]
