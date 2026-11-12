"""Exact minimum-mask MILP for problem P_eps, solved with HiGHS and iterative constraint generation.

Variables: m_i in {0, 1} for i in D_u(t) (1 = masked). Objective: minimise sum_i m_i. Masked scores are linear:
s_M(j) = s(j) - sum_i A[i, j] m_i, so every ordering "a above b by eps" is a linear row
(s(a) - s(b)) - sum_i (A[i, a] - A[i, b]) m_i >= eps.

C1 without C2 (settings A, A3)
    y_c in {0, 1} for each candidate c that can end above t; y_c = 1 forces s_M(c) >= s_M(t) + eps (big-M);
    sum_c y_c >= K - #(candidates above t by eps under every mask).
C1 + C2 (settings B, C)
    exactly one replacement outsider r enters: z_r in {0, 1}, sum_r z_r = 1, z_r = 1 forces
    s_M(r) >= s_M(t) + eps. Every retained j: s_M(j) >= s_M(t) + eps, and s_M(j) >= s_M(c) + eps for every
    outsider c (relaxed by big-M when c is the chosen replacement).
C3 (settings A3, C)
    for every retained j whose qualifying origins all lie in D_u(t): sum_{i in Q_j} (1 - m_i) >= 1.

Constraint generation. HiGHS has no lazy-constraint callback in its Python interface, so the dense
"retained above outsider c" rows are generated iteratively: the first model contains them for the THREAT_INIT
outsiders with the highest unmasked scores; after each solve, every outsider c that would end within eps of a
retained item under the returned mask is added, and the model is rebuilt and re-solved. Every intermediate model is
a relaxation of the full model, so (i) its infeasibility proves infeasibility, (ii) its optimal value is a lower
bound, and (iii) the loop stops with a proven optimum as soon as the returned mask satisfies all constraints.

Replacement pruning: an outsider r can only be the replacement if s(r) >= max_c lb(c) over all outsiders c, where
lb(c) = s(c) - sum_{i in D} A[i, c] is c's lowest possible masked score. Proof: if a mask is feasible with
replacement r', it is also feasible with the best outsider b under the masked scores; then
s(b) >= s_M(b) >= s_M(c) >= lb(c) for every outsider c. The rule never removes r itself (s(r) >= lb(r)).

Outcomes: ``optimal`` (verified mask, matching lower bound), ``feasible`` (verified mask, optimality not proven
within the time budget), ``infeasible`` (proven for P_eps), ``timeout`` (neither). The time budget covers model
building and all solves. Masks returned by HiGHS are re-checked in NumPy and accepted when every needed gap is at
least eps - FEAS_TOL (the solver's feasibility tolerance); every returned mask is additionally verified under the
exact problem P by :mod:`epic_rec.evaluation.verify`.
"""

from __future__ import annotations

import time
from typing import Any

import highspy
import numpy as np

from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting

THREAT_INIT = 40
FEAS_TOL = 1e-9  # HiGHS primal / MIP feasibility tolerance; accepted masks have every needed gap >= eps - FEAS_TOL


def replacement_candidates(inst: ExplanationInstance, lb: np.ndarray, eps: float = EPS, prune: bool = True
                           ) -> np.ndarray:
    O, t, s = inst.outsiders, inst.t, inst.s
    keep = s[O] >= lb[t] + eps
    if prune and len(O):
        keep &= s[O] >= lb[O].max() - 1e-12
    return O[keep]


class _Model:
    def __init__(self, n: int, time_limit: float, presolve: str, threads: int = 1):
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("time_limit", float(max(time_limit, 0.01)))
        self.h.setOptionValue("threads", threads)
        self.h.setOptionValue("presolve", presolve)
        self.h.setOptionValue("mip_rel_gap", 0.0)
        self.h.setOptionValue("mip_abs_gap", 0.999)  # the objective is integer
        self.h.setOptionValue("primal_feasibility_tolerance", 1e-9)
        self.h.setOptionValue("mip_feasibility_tolerance", 1e-9)
        self.nv = 0
        for _ in range(n):
            self.binary(1.0)

    def binary(self, cost: float = 0.0) -> int:
        self.h.addVar(0, 1)
        self.h.changeColIntegrality(self.nv, highspy.HighsVarType.kInteger)
        if cost:
            self.h.changeColCost(self.nv, cost)
        self.nv += 1
        return self.nv - 1

    def ge(self, const: float, coef: np.ndarray, rhs: float, extra: Any = ()) -> None:
        """const + coef . m + sum(extra) >= rhs."""
        idx = list(np.flatnonzero(coef))
        val = list(coef[idx])
        for k, v in extra:
            idx.append(k)
            val.append(v)
        self.h.addRow(rhs - const, highspy.kHighsInf, len(idx), np.array(idx, dtype=np.int32),
                      np.array(val, dtype=np.float64))

    def row(self, lo: float, hi: float, idx: list[int], val: list[float]) -> None:
        self.h.addRow(lo, hi, len(idx), np.array(idx, dtype=np.int32), np.array(val, dtype=np.float64))


def solve_milp(inst: ExplanationInstance, setting: str | Setting, eps: float = EPS, time_limit: float = 20.0,
               prune: bool = True, threat_init: int = THREAT_INIT, threads: int = 1) -> dict[str, Any]:
    """Minimum |M| with M subset of D_u(t) for P_eps. Returns a dict with status, size, mask (item ids), bound,
    gap, time, iterations and the number of generated threat outsiders."""
    if eps <= 0:
        raise ValueError("the MILP needs eps > 0; problem P (eps = 0) is checked by verification and enumeration")
    st = get_setting(setting)
    t0 = time.time()
    t, sc, s = inst.t, inst.scope, inst.s
    n = len(sc)
    Asc = inst.A[inst.rows(sc)] if n else np.zeros((0, len(s)))
    lb = s - Asc.sum(0)
    elig = np.flatnonzero(inst.eligible)
    best: dict[str, Any] = dict(M=None, size=None)
    state = dict(lower=0.0, iters=0, threats=0)

    def remaining() -> float:
        return time_limit - (time.time() - t0)

    def out(status: str) -> dict[str, Any]:
        el = time.time() - t0
        base = dict(time=el, iterations=state["iters"], threats=state["threats"], eps=eps, setting=st.name)
        if status == "infeasible":
            return dict(status="infeasible", size=None, mask=None, bound=None, gap=None, **base)
        if best["M"] is not None and (status == "optimal" or best["size"] <= np.ceil(state["lower"] - 1e-9)):
            return dict(status="optimal", size=best["size"], mask=best["M"], bound=best["size"], gap=0.0, **base)
        bound = float(np.ceil(state["lower"] - 1e-9))
        if best["M"] is not None:
            return dict(status="feasible", size=best["size"], mask=best["M"], bound=bound,
                        gap=(best["size"] - bound) / max(best["size"], 1), **base)
        return dict(status="timeout", size=None, mask=None, bound=bound, gap=None, **base)

    def offer(M: list[int] | None) -> None:
        # a solver mask is accepted only if it satisfies P_eps up to the solver tolerance, re-checked in NumPy
        if M is not None and inst.check(M, st, margin=eps - FEAS_TOL) and (best["M"] is None or len(M) < best["size"]):
            best["M"], best["size"] = sorted(int(i) for i in M), len(M)

    def diff(a: int, b: int) -> tuple[float, np.ndarray]:
        return s[a] - s[b], -(Asc[:, a] - Asc[:, b])

    def bigM(const: float, coef: np.ndarray, rhs: float) -> float:
        return max(0.0, rhs - (const + np.minimum(coef, 0).sum())) + 1e-9

    def solve(m: _Model) -> tuple[Any, Any, Any, list[int] | None]:
        m.h.setOptionValue("time_limit", float(max(remaining(), 0.01)))
        m.h.run()
        ms, info = m.h.getModelStatus(), m.h.getInfo()
        sol = m.h.getSolution().col_value if info.primal_solution_status == 2 else None
        M = [int(sc[k]) for k in range(n) if sol[k] > 0.5] if sol is not None else None
        return ms, info, sol, M

    def add_c3(m: _Model) -> None:
        vpos = {int(i): k for k, i in enumerate(sc)}
        for q in inst.Q_free:
            if q is not None:
                idx = [vpos[int(i)] for i in q]
                m.row(-highspy.kHighsInf, len(idx) - 1, idx, [1.0] * len(idx))

    MS = highspy.HighsModelStatus
    if st.c3 and any(q is not None and len(q) == 0 for q in inst.Q_free):
        return out("infeasible")  # a retained item has no qualifying evidence at all

    if not st.c2:
        others = elig[elig != t]
        always = others[lb[others] >= s[t] + eps]
        need = inst.K - len(always)
        if need <= 0 and (not st.c3 or inst.c3([])):
            best["M"], best["size"] = [], 0
            return out("optimal")
        cand = others[(s[others] >= lb[t] + eps) & (lb[others] < s[t] + eps)]
        if len(cand) < need:
            return out("infeasible")
        m = _Model(n, remaining(), "on", threads)
        ys = []
        for c in cand:
            y = m.binary()
            ys.append(y)
            const, coef = diff(int(c), t)
            B = bigM(const, coef, eps)
            m.ge(const, coef, eps - B, extra=[(y, -B)])
        if need > 0:
            m.row(need, highspy.kHighsInf, ys, [1.0] * len(ys))
        if st.c3:
            add_c3(m)
        ms, info, sol, M = solve(m)
        state["iters"] = 1
        if ms == MS.kInfeasible:
            return out("infeasible")
        offer(M)
        if np.isfinite(info.mip_dual_bound):
            state["lower"] = max(state["lower"], info.mip_dual_bound)
        if ms == MS.kOptimal and best["M"] is not None and best["size"] == len(M):
            return out("optimal")
        return out("timeout")

    R = inst.retained
    repl = replacement_candidates(inst, lb, eps, prune)
    if len(repl) == 0:
        return out("infeasible")
    O = inst.outsiders
    threat = set(O[np.argsort(-s[O], kind="stable")][:threat_init].tolist())
    while True:
        if remaining() <= 0:
            return out("timeout")
        m = _Model(n, remaining(), "off", threads)
        state["iters"] += 1
        state["threats"] = len(threat)
        z: dict[int, int] = {}
        for r in repl:
            z[int(r)] = m.binary()
            const, coef = diff(int(r), t)
            B = bigM(const, coef, eps)
            m.ge(const, coef, eps - B, extra=[(z[int(r)], -B)])
        m.row(1, 1, list(z.values()), [1.0] * len(z))
        for j in R:
            const, coef = diff(int(j), t)
            m.ge(const, coef, eps)
            for c in threat:
                if s[c] + eps > lb[j]:  # otherwise j beats c by eps under every mask
                    const, coef = diff(int(j), int(c))
                    m.ge(const, coef, eps, extra=[(z[c], bigM(const, coef, eps))] if c in z else ())
        if st.c3:
            add_c3(m)
        ms, info, sol, M = solve(m)
        if ms == MS.kInfeasible:
            return out("infeasible")
        if np.isfinite(info.mip_dual_bound):
            state["lower"] = max(state["lower"], info.mip_dual_bound)
        offer(M)
        if ms != MS.kOptimal:
            return out("timeout")
        state["lower"] = max(state["lower"], len(M))
        if best["M"] is not None and best["size"] <= len(M):
            return out("optimal")
        sM = inst.masked_scores(M)
        rsel = {r for r in z if sol[z[r]] > 0.5}
        viol = [int(c) for c in O if c not in threat and c not in rsel and (sM[c] + eps > sM[R]).any()]
        if not viol:
            return out("timeout")  # cannot happen when the model and check() agree; reported, not hidden
        threat.update(viol)
