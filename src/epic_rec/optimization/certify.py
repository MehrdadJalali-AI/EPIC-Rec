"""Independent certification of MILP optimality and infeasibility claims.

A MILP claim rests on one solver run. On one MovieLens pair the study configuration of HiGHS (presolve off,
feasibility tolerances 1e-9) reported a non-minimal mask as optimal, although the model admits a smaller one. Every
claim is therefore re-checked with a second configuration, presolve on with HiGHS's standard tolerances
(:data:`CERT_OPTIONS`), on a model that only asks the question that would refute the claim:

optimal, size k > 0
    solve with |M| <= k - 1. Infeasible confirms the optimum. A returned mask replaces the result only if it passes
    :func:`epic_rec.evaluation.verify.verify` (problems P and P_eps); the smaller size is then certified again.
    Anything else leaves the claim unconfirmed and the result is downgraded to ``feasible`` (verified mask,
    optimality open).
infeasible
    solve without a size cap. Infeasible confirms. A verified mask refutes the claim; the new result is ``feasible``
    and certified like an optimum. Anything else downgrades the result to ``timeout``.

Optima of size 0 and infeasibility found by the structural pre-checks of :func:`solve_milp` (no admissible
replacement, a retained item without qualifying evidence, fewer candidates than needed) involve no search and
are labelled ``structural``.
"""

from __future__ import annotations

import time
from typing import Any

from epic_rec.evaluation.verify import verify
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting
from epic_rec.optimization.isolated import run_isolated
from epic_rec.optimization.milp import solve_milp

CERT_OPTIONS = dict(presolve="on", primal_tol=1e-7, mip_tol=1e-6)


def _verified(inst: ExplanationInstance, x: dict, st: Setting, eps: float) -> bool:
    return x.get("mask") is not None and bool(verify(inst, x["mask"], st, eps=eps)["passed"])


def certify(inst: ExplanationInstance, setting: str | Setting, raw: dict[str, Any], eps: float = EPS,
            time_limit: float = 60.0, max_rounds: int = 5) -> dict[str, Any]:
    """Return ``raw`` (a :func:`solve_milp` result) with a ``certification`` field and, if the claim is refuted or
    cannot be confirmed, the corrected status, size, mask, bound and gap. ``original_status`` and
    ``original_size`` keep the uncertified claim; ``certify_time`` is the time spent here."""
    st = get_setting(setting)
    t0 = time.time()
    out = dict(raw, original_status=raw["status"], original_size=raw.get("size"))

    def done(**kw: Any) -> dict[str, Any]:
        out.update(kw, certify_time=time.time() - t0)
        return out

    if raw["status"] not in ("optimal", "infeasible"):
        return done(certification="not_applicable")
    if raw["status"] == "optimal" and raw["size"] == 0:
        return done(certification="structural")
    if raw["status"] == "infeasible" and not raw.get("iterations"):
        return done(certification="structural")

    refuted = False
    if raw["status"] == "infeasible":
        x = solve_milp(inst, st, eps=eps, time_limit=time_limit, **CERT_OPTIONS)
        if x["status"] == "infeasible":
            return done(certification="confirmed")
        if not _verified(inst, x, st, eps):
            return done(status="timeout", size=None, mask=None, bound=None, gap=None, certification="undecided")
        refuted = True
        out.update(status="feasible", size=x["size"], mask=x["mask"])

    for _ in range(max_rounds):
        x = solve_milp(inst, st, eps=eps, time_limit=time_limit, max_size=out["size"] - 1, **CERT_OPTIONS)
        if x["status"] == "infeasible":
            return done(status="optimal", bound=out["size"], gap=0.0,
                        certification="corrected" if refuted else "confirmed")
        if not _verified(inst, x, st, eps):
            break
        refuted = True
        out.update(status="feasible", size=x["size"], mask=x["mask"])
        if out["size"] == 0:
            return done(status="optimal", bound=0, gap=0.0, certification="corrected")
    return done(status="feasible", bound=None, gap=None,
                certification="corrected_undecided" if refuted else "undecided")


def certify_isolated(inst: ExplanationInstance, setting: str | Setting, raw: dict[str, Any], eps: float = EPS,
                     time_limit: float = 60.0, max_rounds: int = 5) -> dict[str, Any]:
    """:func:`certify` in a child process with a hard wall-clock limit; a killed certification is ``undecided``."""
    if (raw["status"] not in ("optimal", "infeasible") or (raw["status"] == "optimal" and raw["size"] == 0)
            or (raw["status"] == "infeasible" and not raw.get("iterations"))):
        return certify(inst, setting, raw, eps=eps, time_limit=time_limit)
    hard = (max_rounds + 1) * time_limit + 40
    c = run_isolated(certify, inst, setting, raw, hard_limit=hard, eps=eps, time_limit=time_limit,
                     max_rounds=max_rounds)
    if not c.get("killed"):
        return c
    out = dict(raw, original_status=raw["status"], original_size=raw.get("size"), certification="undecided",
               certify_time=hard)
    if raw["status"] == "infeasible":
        out.update(status="timeout", size=None, mask=None, bound=None, gap=None)
    else:
        out.update(status="feasible", bound=None, gap=None)
    return out
