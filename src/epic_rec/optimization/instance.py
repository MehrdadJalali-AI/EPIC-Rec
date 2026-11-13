"""One EPIC-Rec problem: user u, list size K, target t, evidence threshold, constraint setting.

Ranking rule (problem P)
------------------------
Candidates C_u (items not in H_u) are ranked by masked score s_M, descending; ties are broken by item id,
ascending. top-K(M) is the first K items. L = top-K(empty mask), t in L, R = L minus t (retained items),
O = C_u minus L (outsiders).

Margin variant (problem P_eps)
------------------------------
Every ordering a setting needs must hold with gap >= eps in masked score:

* C1 (no C2): at least K candidates c != t with s_M(c) >= s_M(t) + eps;
* C1+C2: some outsider r with s_M(r) >= s_M(t) + eps, and every j in R with s_M(j) >= s_M(t) + eps and
  s_M(j) >= s_M(c) + eps for every other outsider c.

Every P_eps-feasible mask is P-feasible; the converse can fail only when a mask needs a gap in [0, eps) or a tie
resolved by item id.

Mask domain: D_u(t) = {i in H_u : A[i, t] > 0} (``scope="target"``) or all of H_u (``scope="history"``).

Evidence (C3)
-------------
The evidence unit is an individual path (see :mod:`epic_rec.models.path_additive`). For a retained item j let
P[i, j] be the largest single-path contribution from interaction i to j, and

    tau_j = max(alpha * max_i P[i, j], phi),   Q_j = {i in H_u : P[i, j] >= tau_j}.

Every path from an origin in Q_j with contribution >= tau_j is a qualifying path; it is *faithful* when disabling
it lowers s_M(j) by at least phi (checked by recomputation in :mod:`epic_rec.evaluation.verify`). C3 holds when
every retained item keeps a qualifying path, i.e. some i in Q_j is unmasked. A retained item with an empty Q_j
("evidence-free") can never satisfy C3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-6


@dataclass(frozen=True)
class Setting:
    """A combination of constraints; C1 is always required."""

    name: str
    c2: bool
    c3: bool

    @property
    def label(self) -> str:
        return "C1" + (" + C2" if self.c2 else "") + (" + C3" if self.c3 else "")


SETTINGS: dict[str, Setting] = {
    "A": Setting("A", c2=False, c3=False),
    "B": Setting("B", c2=True, c3=False),
    "C": Setting("C", c2=True, c3=True),
    "A3": Setting("A3", c2=False, c3=True),  # ablation "without C2": C1 + C3
}


def get_setting(s: str | Setting) -> Setting:
    if isinstance(s, Setting):
        return s
    aliases = {"C1": "A", "C1+C2": "B", "C1+C2+C3": "C", "C1+C3": "A3"}
    key = aliases.get(s.replace(" ", ""), s)
    return SETTINGS[key]


class ExplanationInstance:
    """Linearised problem for one user: masked scores s_M = s - sum_{i in M} A[i]."""

    def __init__(self, model: Any, u: int, K: int = 10, alpha: float = 0.5, phi: float = 0.0,
                 scope: str = "target"):
        H = model.history(u)
        s = model.score(u)
        eligible = model.candidates(u)
        self._setup(H, model.contributions(u), model.max_path(u), s, eligible, K, alpha, phi, scope, u)
        self.model = model

    @classmethod
    def from_arrays(cls, A: np.ndarray, P: np.ndarray, s: np.ndarray, K: int, alpha: float = 0.5,
                    phi: float = 0.0, eligible: np.ndarray | None = None, scope: str = "target"
                    ) -> ExplanationInstance:
        """Synthetic instance: A, P are |H| x n (history rows are interactions 0..|H|-1), s has length n."""
        self = cls.__new__(cls)
        n = len(s)
        self._setup(np.arange(A.shape[0]), np.asarray(A, float), np.asarray(P, float), np.asarray(s, float),
                    np.ones(n, bool) if eligible is None else np.asarray(eligible, bool), K, alpha, phi, scope, -1)
        self.model = None
        return self

    def _setup(self, H, A, P, s, eligible, K, alpha, phi, scope, u) -> None:
        self.u, self.K, self.alpha, self.phi, self.scope_mode = int(u), int(K), float(alpha), float(phi), scope
        self.H = np.asarray(H, dtype=np.int64)
        self.pos = {int(i): k for k, i in enumerate(self.H)}
        self.A = np.asarray(A, dtype=np.float64)
        self.P = np.asarray(P, dtype=np.float64)
        self.eligible = np.asarray(eligible, bool)
        self.s = np.where(self.eligible, np.asarray(s, dtype=np.float64), -np.inf)
        self.L = self.topk(self.s)
        self.t: int | None = None

    # ------------------------------------------------------------------ ranking
    def topk(self, s: np.ndarray, K: int | None = None) -> np.ndarray:
        idx = np.flatnonzero(self.eligible)
        return idx[np.lexsort((idx, -s[idx]))[: K or self.K]]

    def rows(self, M: Any) -> list[int]:
        """Row indices in A / P of the interactions (item ids) in M."""
        return [self.pos[int(i)] for i in M]

    def masked_scores(self, M: Any) -> np.ndarray:
        r = self.rows(M)
        return self.s - self.A[r].sum(0) if r else self.s.copy()

    # ------------------------------------------------------------------ target
    def set_target(self, t: int) -> ExplanationInstance:
        t = int(t)
        if t not in set(self.L.tolist()):
            raise ValueError(f"target {t} is not in the top-{self.K} list")
        self.t = t
        self.rank = int(np.flatnonzero(self.L == t)[0]) + 1
        self.retained = np.array([j for j in self.L if j != t], dtype=np.int64)
        in_L = np.zeros(len(self.s), bool)
        in_L[self.L] = True
        self.outsiders = np.flatnonzero(self.eligible & ~in_L)
        target_rows = np.flatnonzero(self.A[:, t] > 0)
        self.scope = self.H[target_rows] if self.scope_mode == "target" else self.H.copy()
        Pr = self.P[:, self.retained]
        mx = Pr.max(0) if len(self.H) else np.zeros(len(self.retained))
        self.tau = np.maximum(self.alpha * mx, self.phi)
        self.Q = [
            self.H[np.flatnonzero(Pr[:, k] >= self.tau[k] - 1e-15)] if mx[k] > 0 and mx[k] >= self.phi
            else np.array([], dtype=np.int64)
            for k in range(len(self.retained))
        ]
        self.evidence_free = int(sum(len(q) == 0 for q in self.Q))
        in_scope = set(self.scope.tolist())
        # an item with a qualifying origin outside the mask domain can never lose its evidence
        self.Q_free: list[np.ndarray | None] = [
            None if (len(q) and any(int(i) not in in_scope for i in q)) else q for q in self.Q
        ]
        return self

    # ------------------------------------------------------------------ constraint checks
    def c3(self, M: Any) -> bool:
        Ms = set(int(i) for i in M)
        return all(q is None or (len(q) > 0 and any(int(i) not in Ms for i in q)) for q in self.Q_free)

    def check(self, M: Any, setting: str | Setting, margin: float = 0.0) -> bool:
        """Does mask M satisfy the setting? margin = 0: problem P (id tie-break); margin > 0: P_eps."""
        st = get_setting(setting)
        s = self.masked_scores(M)
        t = self.t
        if margin <= 0:
            top = set(self.topk(s).tolist())
            ok = t not in top
            if st.c2:
                ok = ok and all(int(j) in top for j in self.retained)
        else:
            others = np.flatnonzero(self.eligible)
            others = others[others != t]
            if not st.c2:
                ok = int((s[others] >= s[t] + margin).sum()) >= self.K
            else:
                so = np.sort(s[self.outsiders])[::-1]
                b1 = so[0] if len(so) else -np.inf
                b2 = so[1] if len(so) > 1 else -np.inf
                mr = s[self.retained].min() if len(self.retained) else np.inf
                ok = b1 >= s[t] + margin and mr >= s[t] + margin and mr >= b2 + margin
        if ok and st.c3:
            ok = self.c3(M)
        return bool(ok)

    # ------------------------------------------------------------------ descriptive statistics
    def kendall_retained(self, M: Any) -> float:
        """Kendall's tau-a between the original and masked order of the retained items."""
        s = self.masked_scores(M)
        a, b = self.s[self.retained], s[self.retained]
        n = len(a)
        if n < 2:
            return 1.0
        x = np.sign(a[:, None] - a[None, :]) * np.sign(b[:, None] - b[None, :])
        iu = np.triu_indices(n, 1)
        conc, disc = (x[iu] > 0).sum(), (x[iu] < 0).sum()
        return float((conc - disc) / max(conc + disc, 1))

    def entanglement(self) -> dict[str, float]:
        """How much the target's support overlaps the retained items' support and evidence."""
        D = set(self.scope.tolist())
        rows = self.rows(self.scope)
        supp = self.A[:, self.retained] > 0
        sup_any = {int(self.H[k]) for k in np.flatnonzero(supp.any(1))}
        ev_any = {int(i) for q in self.Q for i in q}
        base = self.s[self.retained]
        from_D = self.A[rows][:, self.retained].sum(0) if rows else np.zeros(len(self.retained))
        return dict(
            domain_size=len(D),
            shared_support=len(D & sup_any),
            shared_support_frac=len(D & sup_any) / max(len(D), 1),
            shared_evidence=len(D & ev_any),
            shared_evidence_frac=len(D & ev_any) / max(len(D), 1),
            retained_score_from_domain=float(np.mean(from_D / np.maximum(base, 1e-12))) if len(base) else 0.0,
        )
