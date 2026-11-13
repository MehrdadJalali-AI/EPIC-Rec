"""Continuous-mask relaxation with discretisation and (optional) verify-and-repair.

Each interaction i in D_u(t) gets a logit w_i and a relaxed mask m_i = sigmoid(w_i) (``relaxation="sigmoid"``)
or a binary-concrete sample m_i = sigmoid((w_i + g_i) / temperature) with logistic noise g_i
(``relaxation="gumbel"``). Because scores are linear in the mask, s(m) = s - m A_D, and the loss is

    loss = hinge_C1 + [C2] sum_j hinge_C2(j) + [C3] sum_j prod_{i in Q_j} m_i + lambda * sum_i m_i,

with hinges on the eps-margins used by P_eps, measured relative to the score gap between rank 1 and rank K + 1.
The product term is the relaxed probability that item j loses all its qualifying evidence. The logits are fitted
with Adam using exact gradients. During optimisation the thresholded mask (m >= 0.5) is checked every
``check_every`` steps and the smallest valid one is kept; at the end the final thresholded mask (or the best valid
one) is returned, optionally passed through :func:`epic_rec.explainers.repair.repair`.

Initialisation: ``init="random"`` draws logits from N(0, 1); ``init="attribution"`` starts from the target-score
attribution (logit = 4 * A[i, t] / max_i A[i, t] - 2), a learned starting point.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from epic_rec.explainers.repair import repair as repair_fn
from epic_rec.optimization.instance import EPS, ExplanationInstance, Setting, get_setting


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def continuous_mask(inst: ExplanationInstance, setting: str | Setting, steps: int = 300, lr: float = 0.1,
                    sparsity: float = 0.02, relaxation: str = "sigmoid", temperature: float = 0.5,
                    init: str = "attribution", repair: bool = True, check_every: int = 10,
                    rng: np.random.Generator | None = None, eps: float = EPS) -> dict[str, Any]:
    st = get_setting(setting)
    rng = rng or np.random.default_rng(0)
    D = np.asarray(inst.scope, dtype=np.int64)
    n = len(D)
    if n == 0:
        return dict(status="not_found", mask=None, size=None, calls=0, details=dict(reason="empty domain"))
    AD = inst.A[inst.rows(D)]
    s0 = inst.s
    t = inst.t
    elig = np.flatnonzero(inst.eligible)
    others = elig[elig != t]
    top_sorted = np.sort(s0[elig])[::-1]
    scale = max(top_sorted[0] - top_sorted[min(inst.K, len(top_sorted) - 1)], 1e-9)
    vpos = {int(i): k for k, i in enumerate(D)}
    Qd = [np.array([vpos[int(i)] for i in q]) for q in inst.Q_free if q is not None and len(q)] if st.c3 else []

    if init == "random":
        w = rng.normal(0.0, 1.0, n)
    elif init == "attribution":
        a = AD[:, t]
        w = 4.0 * a / max(a.max(), 1e-12) - 2.0
    else:
        raise ValueError(f"unknown init {init}")
    mom, vel = np.zeros(n), np.zeros(n)
    best: list[int] | None = None
    calls = 0

    def hinge_grad(a: int, b: int, s: np.ndarray) -> tuple[float, np.ndarray]:
        """hinge(s_a - s_b + eps) / scale and its gradient in m (s_k = s0_k - m . AD[:, k])."""
        v = (s[a] - s[b] + eps) / scale
        if v <= 0:
            return 0.0, np.zeros(n)
        return v, (-AD[:, a] + AD[:, b]) / scale

    for step in range(steps):
        noise = rng.logistic(size=n) if relaxation == "gumbel" else 0.0
        tau = temperature if relaxation == "gumbel" else 1.0
        m = _sigmoid((w + noise) / tau)
        s = s0 - m @ AD
        g = np.zeros(n)
        if not st.c2:
            kth = others[np.argsort(-s[others], kind="stable")[inst.K - 1]]
            g += hinge_grad(t, int(kth), s)[1]
        else:
            O = inst.outsiders
            order = O[np.argsort(-s[O], kind="stable")]
            r = int(order[0])
            o2 = int(order[1]) if len(order) > 1 else None
            g += hinge_grad(t, r, s)[1]
            for j in inst.retained:
                g += hinge_grad(t, int(j), s)[1]
                if o2 is not None:
                    g += hinge_grad(o2, int(j), s)[1]
        for q in Qd:
            prod = np.prod(m[q])
            g[q] += prod / np.maximum(m[q], 1e-12)
        g += sparsity
        gw = g * m * (1 - m) / tau
        mom = 0.9 * mom + 0.1 * gw
        vel = 0.999 * vel + 0.001 * gw**2
        w -= lr * (mom / (1 - 0.9 ** (step + 1))) / (np.sqrt(vel / (1 - 0.999 ** (step + 1))) + 1e-8)
        if (step + 1) % check_every == 0:
            cand = [int(D[k]) for k in np.flatnonzero(_sigmoid(w) >= 0.5)]
            calls += 1
            if inst.check(cand, st, eps) and (best is None or len(cand) < len(best)):
                best = sorted(cand)
    m_final = _sigmoid(w)
    final = sorted(int(D[k]) for k in np.flatnonzero(m_final >= 0.5))
    calls += 1
    if inst.check(final, st, eps) and (best is None or len(final) <= len(best)):
        best = final
    priority = {int(D[k]): float(m_final[k]) for k in range(n)}
    details: dict[str, Any] = dict(init=init, relaxation=relaxation, steps=steps, discretised_size=len(final),
                                   discretised_valid=bool(inst.check(final, st, eps)))
    if best is not None and not repair:
        return dict(status="found", mask=best, size=len(best), calls=calls, details=details)
    if not repair:
        return dict(status="not_found", mask=None, size=None, calls=calls, details=details)
    start = best if best is not None else final
    rep = repair_fn(inst, start, st, priority=priority, eps=eps)
    details["repair"] = {k: (len(v) if isinstance(v, list) else v) for k, v in rep.items() if k != "mask"}
    if rep["success"]:
        return dict(status="found", mask=rep["mask"], size=len(rep["mask"]), calls=calls, details=details)
    return dict(status="not_found", mask=None, size=None, calls=calls, details=details)
