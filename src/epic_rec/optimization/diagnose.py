"""Infeasibility reports: why a target cannot be removed under a setting.

The MILP's ``infeasible`` status is the proof. The report adds (i) quick certificates that prove infeasibility
without a solver when they apply, (ii) the outcome of weaker settings, which locates the constraint family that
causes the conflict, and (iii) entanglement statistics. Weaker-setting outcomes are passed in when the caller has
already solved them (as the experiment runner does) or solved here otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting
from epic_rec.optimization.milp import replacement_candidates, solve_milp


def certificates(inst: ExplanationInstance, setting: str | Setting, eps: float = EPS) -> list[dict[str, Any]]:
    """Sufficient conditions for P_eps infeasibility that need no solver."""
    st = get_setting(setting)
    t, s = inst.t, inst.s
    rows = inst.rows(inst.scope)
    lb = s - (inst.A[rows].sum(0) if rows else 0.0)
    out = []
    elig = np.flatnonzero(inst.eligible)
    others = elig[elig != t]
    reach = int((s[others] >= lb[t] + eps).sum())
    if reach < inst.K:
        out.append(dict(kind="c1_unreachable", detail=f"only {reach} candidates can end above t even when every "
                        f"interaction in D_u(t) is masked; K = {inst.K} are needed"))
    if st.c2 and len(replacement_candidates(inst, lb, eps)) == 0:
        out.append(dict(kind="no_replacement", detail="no outsider can end above t while staying the best outsider"))
    if st.c2:
        stuck = [int(j) for j in inst.retained if s[j] < lb[t] + eps]
        if stuck:
            out.append(dict(kind="retained_cannot_stay_above_t", items=stuck,
                            detail="these retained items stay below t's lowest possible score"))
    if st.c3:
        free = [int(inst.retained[k]) for k, q in enumerate(inst.Q_free) if q is not None and len(q) == 0]
        if free:
            out.append(dict(kind="evidence_free_items", items=free,
                            detail="retained items without any qualifying evidence path"))
    return out


def evidence_blocking_items(inst: ExplanationInstance) -> list[int]:
    """Retained items whose every qualifying origin lies in the target's support D_u(t).

    These are the items for which C3 can conflict with removing t (necessary, not sufficient, for a C3 conflict).
    """
    return [int(inst.retained[k]) for k, q in enumerate(inst.Q_free) if q is not None and len(q) > 0]


def infeasibility_report(inst: ExplanationInstance, setting: str | Setting, eps: float = EPS,
                         outcomes: dict[str, str] | None = None, time_limit: float = 10.0) -> dict[str, Any]:
    st = get_setting(setting)
    outcomes = dict(outcomes or {})
    ladder = ["A"] + (["B"] if st.c2 else []) + (["A3"] if st.c3 else [])
    for name in ladder:
        if name != st.name and name not in outcomes:
            outcomes[name] = solve_milp(inst, name, eps=eps, time_limit=time_limit)["status"]
    feas = {k: v in ("optimal", "feasible") for k, v in outcomes.items()}
    free = "evidence-free item: a retained item has no qualifying path even before masking"
    if outcomes.get("A") == "infeasible":
        cause = "C1 alone is infeasible: the target cannot leave the list by masking its support"
    elif st.c2 and st.c3:
        if feas.get("B") and feas.get("A3"):
            cause = "joint conflict: C1 + C2 and C1 + C3 are each feasible, but not together"
        elif feas.get("B") and inst.evidence_free:
            cause = free
        elif feas.get("B"):
            cause = "evidence entanglement: C1 + C2 is feasible, adding C3 makes it infeasible"
        elif outcomes.get("B") == "infeasible":
            cause = "list entanglement: C1 + C2 is already infeasible"
        else:
            cause = "undetermined: a weaker setting timed out"
    elif st.c2:
        cause = "list entanglement: removing t forces another retained item out of the list"
    elif st.c3 and inst.evidence_free:
        cause = free
    elif st.c3:
        cause = "evidence entanglement: removing t removes the last qualifying evidence of a retained item"
    else:
        cause = "C1 alone is infeasible"
    return dict(
        setting=st.name,
        cause=cause,
        weaker_settings=outcomes,
        certificates=certificates(inst, st, eps),
        evidence_blocking_items=evidence_blocking_items(inst) if st.c3 else [],
        entanglement=inst.entanglement(),
    )
