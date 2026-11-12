"""Explanation objects and the single entry point :func:`explain`.

Every method's output is independently verified (:func:`epic_rec.evaluation.verify.verify`). ``success`` is
True only when the verifier confirms every requested constraint; a candidate that fails verification is
reported with ``status="verification_failed"`` and never counted as a success.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from epic_rec.evaluation.verify import verify
from epic_rec.explainers.beam import beam_search
from epic_rec.explainers.continuous import continuous_mask
from epic_rec.explainers.greedy import greedy
from epic_rec.optimization.diagnose import infeasibility_report
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting
from epic_rec.optimization.isolated import run_isolated
from epic_rec.optimization.milp import solve_milp

EXACT_STATUSES = ("optimal", "feasible", "infeasible", "timeout")


@dataclass
class Explanation:
    """Result of explaining the removal of ``target`` from ``user``'s top-K list.

    status       exact: optimal | feasible | infeasible | timeout; approximate: found | not_found;
                 any method: verification_failed
    mask         item ids of the masked history interactions (None if no mask)
    success      True iff a mask exists and the independent verifier confirmed every requested constraint
    bound, gap   MILP lower bound on the minimum size and relative gap (exact method only)
    verification output of :func:`epic_rec.evaluation.verify.verify` (None if no mask)
    infeasibility report from :func:`epic_rec.optimization.diagnose.infeasibility_report` (exact, infeasible)
    """

    user: int
    target: int
    rank: int
    K: int
    setting: str
    setting_label: str
    method: str
    eps: float
    alpha: float
    phi: float
    status: str
    mask: list[int] | None
    size: int | None
    success: bool
    runtime: float
    bound: float | None = None
    gap: float | None = None
    verification: dict[str, Any] | None = None
    infeasibility: dict[str, Any] | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        head = f"user {self.user}, target {self.target} (rank {self.rank}/{self.K}), {self.setting_label}, " \
               f"{self.method}: {self.status}"
        if self.mask is not None:
            v = self.verification or {}
            head += f"; mask {self.mask} (|M| = {self.size}); verified: {self.success}"
            if v:
                head += f"; list overlap {v['overlap']:.2f}, Kendall tau {v['kendall_retained']:.2f}"
                if v.get("evidence_survival") is not None:
                    head += f", evidence survival {v['evidence_survival']:.2f}"
        if self.infeasibility:
            head += f"; cause: {self.infeasibility['cause']}"
        return head


def _finish(inst: ExplanationInstance, st: Setting, method: str, eps: float, raw: dict[str, Any], t0: float,
            exact: bool, evidence: bool) -> Explanation:
    mask = raw.get("mask")
    ver = verify(inst, mask, st, eps=eps, with_evidence=evidence or st.c3) if mask is not None else None
    status = raw["status"]
    success = bool(ver and ver["passed"])
    if mask is not None and not success:
        status = "verification_failed"
    details = {k: v for k, v in raw.items() if k not in ("status", "mask", "size", "bound", "gap")}
    return Explanation(
        user=inst.u, target=int(inst.t), rank=inst.rank, K=inst.K, setting=st.name, setting_label=st.label,
        method=method, eps=eps, alpha=inst.alpha, phi=inst.phi, status=status, mask=mask,
        size=raw.get("size") if mask is not None else None, success=success, runtime=time.time() - t0,
        bound=raw.get("bound") if exact else None, gap=raw.get("gap") if exact else None,
        verification=ver, details=details,
    )


def explain(inst: ExplanationInstance, setting: str | Setting = "C", method: str = "exact", eps: float = EPS,
            time_limit: float = 20.0, isolate: bool = False, diagnose: bool = True, evidence: bool = True,
            rng: np.random.Generator | None = None, **kwargs: Any) -> Explanation:
    """Explain the instance's current target (``inst.set_target`` must have been called).

    method: ``exact`` (MILP), ``beam``, ``greedy:<criterion>[:c1_stop]``, ``continuous`` (keyword arguments are
    passed on, e.g. ``init``, ``relaxation``, ``repair``).
    """
    st = get_setting(setting)
    t0 = time.time()
    if method == "exact":
        if isolate:
            raw = run_isolated(solve_milp, inst, st, hard_limit=time_limit + 40, eps=eps, time_limit=time_limit,
                               **kwargs)
        else:
            raw = solve_milp(inst, st, eps=eps, time_limit=time_limit, **kwargs)
        ex = _finish(inst, st, method, eps, raw, t0, exact=True, evidence=evidence)
        if raw["status"] == "infeasible" and diagnose:
            ex.infeasibility = infeasibility_report(inst, st, eps=eps, time_limit=time_limit)
        return ex
    if method == "beam":
        raw = beam_search(inst, st, eps=eps, **kwargs)
    elif method.startswith("greedy"):
        parts = method.split(":")
        criterion = parts[1] if len(parts) > 1 else "violation"
        mode = parts[2] if len(parts) > 2 else "constrained"
        raw = greedy(inst, st, criterion=criterion, mode=mode, rng=rng, eps=eps, **kwargs)
        if mode == "c1_stop" and raw["mask"] is not None:
            # report which constraints the C1-only mask happens to satisfy, without counting it as a success
            # for the stricter setting
            ver = verify(inst, raw["mask"], "A", eps=eps, with_evidence=True)
            return Explanation(
                user=inst.u, target=int(inst.t), rank=inst.rank, K=inst.K, setting=st.name, setting_label=st.label,
                method=method, eps=eps, alpha=inst.alpha, phi=inst.phi, status=raw["status"], mask=raw["mask"],
                size=raw["size"], success=bool(ver["passed"] and (ver["c2"] and ver["c1_c2_margin"] or not st.c2)
                                               and (ver["c3"] or not st.c3)),
                runtime=time.time() - t0, verification=ver, details=dict(calls=raw["calls"], mode="c1_stop"))
    elif method == "continuous":
        raw = continuous_mask(inst, st, rng=rng, eps=eps, **kwargs)
    else:
        raise ValueError(f"unknown method {method}")
    return _finish(inst, st, method, eps, raw, t0, exact=False, evidence=evidence)
