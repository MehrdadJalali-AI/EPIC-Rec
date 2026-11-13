"""Independent verification of a mask.

The verifier does not reuse the solver's matrices. With a model attached to the instance it

1. recomputes all masked scores with ``model.score(u, M)`` (sparse row sums of W over the unmasked history) and
   re-ranks the candidates with the id tie-break (problem P), and separately checks the margin variant P_eps;
2. recomputes the target's and every retained item's masked score by summing individual paths
   (``model.score_from_paths``) and requires agreement with step 1 (score decomposition check);
3. for C3, enumerates the surviving paths of every retained item, keeps the qualifying ones
   (origin unmasked, contribution >= tau_j) and tests faithfulness: disabling a path must lower the item's
   recomputed score by at least ``phi`` (and by a positive amount). An item keeps its evidence if at least one
   qualifying path is faithful.

A mask counts as a success (``passed``) only if every requested constraint passes here under *both* rankings:
problem P (exact scores, id tie-break) and problem P_eps (every required gap >= eps, accepted down to
eps - 1e-9 for solver round-off). The exact solver solves P_eps, so all methods are held to the same problem;
``passed_P`` alone is reported to quantify the difference between the two semantics. Without a model (synthetic array
instances) the checks fall back to the instance's linear algebra and the report says ``independent: False``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting
from epic_rec.optimization.milp import FEAS_TOL
from epic_rec.paths.evidence import evidence_report

DECOMP_TOL = 1e-9


def _p_eps_ok(s: np.ndarray, inst: ExplanationInstance, st: Setting, eps: float) -> bool:
    t = inst.t
    elig = np.flatnonzero(inst.eligible)
    others = elig[elig != t]
    if not st.c2:
        return int((s[others] >= s[t] + eps).sum()) >= inst.K
    so = np.sort(s[inst.outsiders])[::-1]
    b1 = so[0] if len(so) else -np.inf
    b2 = so[1] if len(so) > 1 else -np.inf
    mr = s[inst.retained].min()
    return bool(b1 >= s[t] + eps and mr >= s[t] + eps and mr >= b2 + eps)


def verify(inst: ExplanationInstance, M: Any, setting: str | Setting, eps: float = EPS,
           with_evidence: bool | None = None) -> dict[str, Any]:
    st = get_setting(setting)
    M = sorted(int(i) for i in M)
    if any(int(i) not in inst.pos for i in M):
        raise ValueError("mask contains items outside the user's history")
    model = inst.model
    if model is not None:
        s = model.score(inst.u, M)
        s = np.where(inst.eligible, s, -np.inf)
        decomp = max(abs(model.score_from_paths(inst.u, int(j), M) - s[j]) for j in [inst.t, *inst.retained])
    else:
        s = inst.masked_scores(M)
        decomp = 0.0
    top = inst.topk(s)
    top_set = set(top.tolist())
    c1 = inst.t not in top_set
    kept = [int(j) for j in inst.retained if int(j) in top_set]
    c2 = len(kept) == len(inst.retained)
    want_ev = st.c3 if with_evidence is None else with_evidence
    ev = evidence_report(inst, M) if want_ev else None
    c3 = all(e["faithful"] for e in ev) if ev is not None else None
    ok_P = c1 and (c2 or not st.c2) and (bool(c3) or not st.c3) and decomp <= DECOMP_TOL
    c1_m = _p_eps_ok(s, inst, get_setting("A"), eps - FEAS_TOL)
    c12_m = _p_eps_ok(s, inst, get_setting("B"), eps - FEAS_TOL)
    ok_eps = (c12_m if st.c2 else c1_m) and (bool(c3) or not st.c3) and decomp <= DECOMP_TOL
    entered = [int(j) for j in top if int(j) not in set(inst.L.tolist())]
    return dict(
        passed=bool(ok_P and ok_eps),
        passed_P=bool(ok_P),
        passed_P_eps=bool(ok_eps),
        c1=bool(c1),
        c2=bool(c2),
        c3=c3,
        c1_margin=bool(c1_m),
        c1_c2_margin=bool(c12_m),
        independent=model is not None,
        decomposition_error=float(decomp),
        top_k_after=[int(j) for j in top],
        entered=entered,
        overlap=len(kept) / max(len(inst.retained), 1),
        kendall_retained=inst.kendall_retained(M),
        evidence=ev,
        evidence_survival=(np.mean([e["faithful"] for e in ev]) if ev else None),
    )
