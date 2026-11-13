"""Verify-and-repair: turn an arbitrary candidate mask into one that satisfies the setting exactly, or fail.

1. Restore: while C3 is violated (setting with C3), unmask the masked qualifying origin of a violated retained
   item with the lowest priority.
2. Add: while the setting is not satisfied, add the interaction from D_u(t) that most reduces the violation
   surrogate (in C3 settings, only additions that keep C3); fail if nothing can be added or ``max_add`` is reached.
3. Prune: try to unmask each masked interaction in increasing priority and keep the removal if the setting still
   holds, so the result is minimal by inclusion.

Every decision uses the exact check of problem P; the caller verifies the final mask independently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.explainers.common import violation
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting


def repair(inst: ExplanationInstance, M: Any, setting: str | Setting, priority: dict[int, float] | None = None,
           max_add: int = 30, eps: float = EPS) -> dict[str, Any]:
    st = get_setting(setting)
    M = [int(i) for i in M]
    pr = priority or {}
    restored, added, pruned = [], [], []
    if st.c3:
        while not inst.c3(M):
            Ms = set(M)
            bad = next(q for q in inst.Q_free if q is not None and len(q) and all(int(i) in Ms for i in q))
            i = min((int(i) for i in bad), key=lambda x: pr.get(x, 0.0))
            M.remove(i)
            restored.append(i)
    while not inst.check(M, st, eps):
        if len(added) >= max_add:
            return dict(success=False, mask=None, restored=restored, added=added, pruned=pruned,
                        reason="max_add reached")
        cand = [int(i) for i in inst.scope if int(i) not in M]
        if st.c3:
            cand = [i for i in cand if inst.c3(M + [i])]
        if not cand:
            return dict(success=False, mask=None, restored=restored, added=added, pruned=pruned,
                        reason="no admissible interaction left")
        vals = [violation(inst, M + [i], st, eps) for i in cand]
        i = cand[int(np.argmin(vals))]
        M.append(i)
        added.append(i)
    for i in sorted(M, key=lambda x: pr.get(x, 0.0)):
        trial = [j for j in M if j != i]
        if inst.check(trial, st, eps):
            M = trial
            pruned.append(i)
    return dict(success=True, mask=sorted(M), restored=restored, added=added, pruned=pruned, reason=None)
