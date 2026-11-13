"""Exhaustive search over all subsets of D_u(t), for small domains; solves P (margin 0) or P_eps."""

from __future__ import annotations

import itertools
from typing import Any

from epic_rec.optimization.instance import ExplanationInstance, Setting


def exhaustive(inst: ExplanationInstance, setting: str | Setting, max_n: int = 12, margin: float = 0.0
               ) -> dict[str, Any] | None:
    """Minimum |M| by enumeration in order of size; None if |D_u(t)| > max_n."""
    sc = [int(i) for i in inst.scope]
    if len(sc) > max_n:
        return None
    for k in range(len(sc) + 1):
        for M in itertools.combinations(sc, k):
            if inst.check(M, setting, margin=margin):
                return dict(status="optimal", size=k, mask=list(M))
    return dict(status="infeasible", size=None, mask=None)
