"""Greedy removal.

Criteria
--------
violation    : dynamic; add the interaction that most reduces the violation surrogate (constraint-aware)
margin       : dynamic; add the interaction maximising A[i, t] - A[i, r], r = item currently at rank K + 1
               (ranking-margin attribution)
target_score : static; descending contribution to the target's score (target-score attribution)
list_aware   : static; target contribution minus the mean contribution to the retained items
popularity   : static control; most popular history items first
random       : static control; random order

Modes
-----
c1_stop      : add interactions until C1 holds and stop (no list or evidence constraint), then report the mask;
               this measures how often an explainer that only targets removal preserves the list by chance
constrained  : stop when the requested setting holds; in settings with C3, skip additions that would remove the
               last qualifying evidence of a retained item
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.explainers.common import first_outside, static_order, violation
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting

CRITERIA = ("violation", "margin", "target_score", "list_aware", "popularity", "random")


def greedy(inst: ExplanationInstance, setting: str | Setting, criterion: str = "violation",
           mode: str = "constrained", max_depth: int = 15, rng: np.random.Generator | None = None,
           eps: float = EPS) -> dict[str, Any]:
    st = get_setting(setting)
    goal = get_setting("A") if mode == "c1_stop" else st
    filter_c3 = mode == "constrained" and st.c3
    M: list[int] = []
    calls = 0
    order = None if criterion in ("violation", "margin") else static_order(inst, criterion, rng)
    while len(M) < max_depth:
        cand = [int(i) for i in inst.scope if int(i) not in M]
        if filter_c3:
            cand = [i for i in cand if inst.c3(M + [i])]
        if not cand:
            break
        if criterion == "violation":
            vals = [violation(inst, M + [i], goal, eps) for i in cand]
            calls += len(cand)
            nxt = cand[int(np.argmin(vals))]
        elif criterion == "margin":
            s = inst.masked_scores(M)
            r = first_outside(inst, s)
            rows = inst.rows(cand)
            gain = inst.A[rows, inst.t] - inst.A[rows, r]
            nxt = cand[int(np.argmax(gain))]
        else:
            allowed = set(cand)
            nxt = next((i for i in order if i in allowed), None)
            if nxt is None:
                break
        M.append(nxt)
        calls += 1
        if inst.check(M, goal, eps):
            return dict(status="found", mask=sorted(M), size=len(M), calls=calls)
    return dict(status="not_found", mask=None, size=None, calls=calls)
