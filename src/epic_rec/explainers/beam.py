"""Beam search over masks (problem P).

Breadth-first by mask size; at each depth the ``width`` masks with the smallest violation surrogate are expanded.
In settings with C3, masks that break C3 are pruned (C3 is anti-monotone: adding interactions cannot restore
evidence). The first mask that satisfies the setting is returned, so the result is minimal only within the beam.
"""

from __future__ import annotations

from typing import Any

from epic_rec.explainers.common import violation
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting


def beam_search(inst: ExplanationInstance, setting: str | Setting, width: int = 5, max_depth: int = 15,
                budget: int | None = None, eps: float = EPS) -> dict[str, Any]:
    st = get_setting(setting)
    sc = [int(i) for i in inst.scope]
    budget = budget or min(width * max(len(sc), 1) * max_depth, 20000)
    calls = 1
    if inst.check((), st, eps):
        return dict(status="found", mask=[], size=0, calls=calls)
    frontier: list[tuple[int, ...]] = [()]
    for depth in range(1, min(max_depth, len(sc)) + 1):
        cands: dict[tuple[int, ...], float] = {}
        for M in frontier:
            for i in sc:
                if i in M:
                    continue
                N = tuple(sorted(M + (i,)))
                if N in cands or (st.c3 and not inst.c3(N)):
                    continue
                calls += 1
                if inst.check(N, st, eps):
                    return dict(status="found", mask=list(N), size=depth, calls=calls)
                cands[N] = violation(inst, N, st, eps)
                if calls >= budget:
                    return dict(status="not_found", mask=None, size=None, calls=calls, budget_exhausted=True)
        if not cands:
            break
        frontier = sorted(cands, key=lambda k: (cands[k], k))[:width]
    return dict(status="not_found", mask=None, size=None, calls=calls)
