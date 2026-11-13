"""Path-additive graph recommender.

Score of candidate item j for user u with training history H_u (h = |H_u|):

    s(u, j) = beta * log(1 + pop_j) + sum_{p in paths(H_u -> j)} c(p)

An individual path p = (i, rho, x, j) starts at a history interaction i, goes through an intermediate node x and
ends at j:

* knowledge-graph path, relation rho, entity x linked to both i and j by rho:   c(p) = theta_rho / (deg_rho(x) h)
* collaborative path, x another user who interacted with both i and j:          c(p) = theta_co / (sqrt(pop_i pop_j) h)

Summing paths gives s(u, j) = beta log(1 + pop_j) + (1/h) sum_{i in H_u} W[i, j] with
W = sum_rho theta_rho S_rho + theta_co S_co. The path weights theta, beta and the graph statistics are fitted on the
training split and then frozen. Masking a set M of the user's interactions removes every path whose origin is in
M and leaves all other path contributions unchanged (h is part of each path's frozen weight), so

    s_M(u, j) = s(u, j) - sum_{i in M} W[i, j] / h,

which is linear in the binary mask. No retraining is involved, and other users' interactions and KG facts are
never modified. Items in H_u are not candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FsPath
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from epic_rec.data.dataset import InteractionDataset
from epic_rec.utils.logging import get_logger
from epic_rec.utils.seeding import rng_for

log = get_logger(__name__)
COLLAB = "co"


@dataclass(frozen=True)
class Path:
    """One evidence path ``origin -relation-> via <-relation- item``.

    ``via_type`` is ``"entity"`` (KG) or ``"user"`` (collaborative); ``contribution`` is the amount the path adds
    to the item's score.
    """

    origin: int
    relation: str
    via_type: str
    via: int
    item: int
    contribution: float


@dataclass(frozen=True)
class PathGroup:
    """Individual paths sharing origin and relation: ``vias[k]`` is the intermediate node of path k and
    ``contributions[k]`` its contribution."""

    origin: int
    relation: str
    via_type: str
    vias: np.ndarray
    contributions: np.ndarray
    item: int

    def __len__(self) -> int:
        return len(self.vias)

    def paths(self) -> list[Path]:
        return [Path(self.origin, self.relation, self.via_type, int(x), self.item, float(c))
                for x, c in zip(self.vias, self.contributions)]


class PathAdditiveRecommender:
    """Recommender whose scores are exact sums of individual path contributions (see module docstring)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        X: sp.csr_matrix,
        incidence: dict[str, sp.csr_matrix],
        theta: dict[str, float],
        beta: float,
        use_collaborative: bool = True,
    ):
        self.n_users, self.n_items = n_users, n_items
        self.X = X.tocsr().astype(np.float64)
        self.XT = self.X.T.tocsr()
        self.incidence = {r: m.tocsr().astype(np.float64) for r, m in incidence.items()}
        self.incidence_T = {r: m.T.tocsr() for r, m in self.incidence.items()}
        self.ent_deg = {r: np.asarray(m.sum(0)).ravel() for r, m in self.incidence.items()}
        self.theta = {k: float(v) for k, v in theta.items()}
        self.beta = float(beta)
        self.use_collaborative = use_collaborative
        self.pop = np.asarray(self.X.sum(0)).ravel().astype(np.float64)
        self.logpop = np.log1p(self.pop)
        self.hist_len = np.asarray(self.X.sum(1)).ravel().astype(np.int64)
        self._build_W()

    # ------------------------------------------------------------------ construction
    @property
    def relations(self) -> list[str]:
        return sorted(self.incidence) + ([COLLAB] if self.use_collaborative else [])

    def relation_matrix(self, r: str) -> sp.csr_matrix:
        """S_r with S_r[i, j] = sum of single-path weights (before theta and 1/h) between items i and j."""
        if r == COLLAB:
            C = (self.XT @ self.X).tocsr()
            C.setdiag(0)
            C.eliminate_zeros()
            inv = 1.0 / np.sqrt(np.maximum(self.pop, 1.0))
            return sp.diags(inv) @ C @ sp.diags(inv)
        B = self.incidence[r]
        deg = self.ent_deg[r]
        S = (B @ sp.diags(1.0 / np.maximum(deg, 1.0)) @ B.T).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        return S

    def _build_W(self) -> None:
        W = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float64)
        for r in self.relations:
            if self.theta.get(r, 0.0) != 0.0:
                W = W + self.theta[r] * self.relation_matrix(r)
        self.W = W.tocsr()
        self._cooc = (self.XT @ self.X).tocsr() if self.use_collaborative else None

    # ------------------------------------------------------------------ scoring
    def history(self, u: int) -> np.ndarray:
        return self.X[u].indices.copy()

    def candidates(self, u: int) -> np.ndarray:
        elig = np.ones(self.n_items, bool)
        elig[self.history(u)] = False
        return elig

    def contributions(self, u: int) -> np.ndarray:
        """A[k, j] = total contribution of history interaction H_u[k] to item j (shape |H_u| x n_items)."""
        H = self.history(u)
        return self.W[H].toarray() / max(len(H), 1)

    def score(self, u: int, mask: Any = ()) -> np.ndarray:
        """Masked scores of all items for user u (history items included; they are not candidates)."""
        H = self.history(u)
        h = max(len(H), 1)
        mask = set(int(i) for i in mask)
        keep = np.array([i for i in H if int(i) not in mask], dtype=np.int64)
        s = np.asarray(self.W[keep].sum(0)).ravel() / h if len(keep) else np.zeros(self.n_items)
        return s + self.beta * self.logpop

    def score_batch(self, users: np.ndarray) -> np.ndarray:
        Xu = self.X[users]
        h = np.maximum(np.asarray(Xu.sum(1)).ravel(), 1.0)
        return np.asarray((Xu @ self.W).todense()) / h[:, None] + self.beta * self.logpop[None, :]

    def topk(self, u: int, K: int, mask: Any = ()) -> np.ndarray:
        """Top-K candidates by masked score, descending; ties broken by item id, ascending."""
        s = self.score(u, mask)
        idx = np.flatnonzero(self.candidates(u))
        return idx[np.lexsort((idx, -s[idx]))[:K]]

    def masking_effect(self, u: int, i: int) -> np.ndarray:
        """Score change of every item when interaction i is masked (always <= 0)."""
        h = max(self.hist_len[u], 1)
        return -self.W[int(i)].toarray().ravel() / h

    # ------------------------------------------------------------------ paths
    def max_path(self, u: int) -> np.ndarray:
        """P[k, j] = largest single-path contribution from history interaction H_u[k] to item j."""
        H = self.history(u)
        h = max(len(H), 1)
        P = np.zeros((len(H), self.n_items))
        for r in sorted(self.incidence):
            th = self.theta.get(r, 0.0)
            if th <= 0:
                continue
            B, BT, deg = self.incidence[r], self.incidence_T[r], self.ent_deg[r]
            for k, i in enumerate(H):
                ents = B[i].indices
                if not len(ents):
                    continue
                starts, ends = BT.indptr[ents], BT.indptr[ents + 1]
                items = np.concatenate([BT.indices[a:b] for a, b in zip(starts, ends)])
                w = np.repeat(th / np.maximum(deg[ents], 1.0), ends - starts)
                np.maximum.at(P[k], items, w)
        if self.use_collaborative and self.theta.get(COLLAB, 0.0) > 0:
            inv = 1.0 / np.sqrt(np.maximum(self.pop, 1.0))
            for k, i in enumerate(H):
                row = self._cooc[i]
                js = row.indices[row.data > 0]
                np.maximum.at(P[k], js, self.theta[COLLAB] * inv[i] * inv[js])
        P[np.arange(len(H)), H] = 0.0
        return P / h

    def path_groups(self, u: int, j: int, mask: Any = (), origins: Any = None) -> list[PathGroup]:
        """Individual paths from the unmasked history of u to item j, grouped by (origin, relation).

        Each group lists its via-nodes and the contribution of each individual path. Collaborative co-users are
        found by intersecting the training interactions of i and j, so this does not use W.
        """
        H = self.history(u) if origins is None else np.asarray(list(origins), dtype=np.int64)
        h = max(self.hist_len[u], 1)
        mask = set(int(i) for i in mask)
        j = int(j)
        out: list[PathGroup] = []
        for i in H:
            i = int(i)
            if i in mask or i == j:
                continue
            for r in sorted(self.incidence):
                th = self.theta.get(r, 0.0)
                if th <= 0:
                    continue
                B = self.incidence[r]
                shared = np.intersect1d(B[i].indices, B[j].indices, assume_unique=True)
                if len(shared):
                    out.append(PathGroup(i, r, "entity", shared, th / (self.ent_deg[r][shared] * h), j))
            if self.use_collaborative and self.theta.get(COLLAB, 0.0) > 0:
                users = np.intersect1d(self.XT[i].indices, self.XT[j].indices, assume_unique=True)
                users = users[users != u]
                if len(users):
                    c = self.theta[COLLAB] / (np.sqrt(self.pop[i] * self.pop[j]) * h)
                    out.append(PathGroup(i, COLLAB, "user", users, np.full(len(users), c), j))
        return out

    def paths(self, u: int, j: int, mask: Any = (), origins: Any = None) -> list[Path]:
        """Every individual path from the unmasked history of u to item j."""
        return [p for g in self.path_groups(u, j, mask, origins) for p in g.paths()]

    def score_from_paths(self, u: int, j: int, mask: Any = (), disabled: Any = ()) -> float:
        """Recompute s_M(u, j) by summing individual paths (independent of W), optionally disabling paths."""
        total = sum(float(g.contributions.sum()) for g in self.path_groups(u, j, mask))
        total -= sum(p.contribution for p in set(disabled) if p.item == int(j) and p.origin not in set(mask))
        return float(total + self.beta * self.logpop[int(j)])

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, ds: InteractionDataset, cfg: dict | None = None) -> PathAdditiveRecommender:
        """Build the graph from the training split and fit theta, beta by leave-one-out BPR (see fit_weights)."""
        cfg = cfg or {}
        X = ds.matrix("train")
        incidence = build_incidence(ds.item_entity, ds.n_items, cfg.get("relations"))
        use_co = cfg.get("collaborative", True)
        model = cls(ds.n_users, ds.n_items, X, incidence, {r: 1.0 for r in incidence} | {COLLAB: 1.0}, 0.0, use_co)
        theta, beta, info = fit_weights(model, cfg)
        model.theta, model.beta, model.fit_info = theta, beta, info
        model._build_W()
        return model

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | FsPath) -> None:
        path = FsPath(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = dict(
            shape=np.array([self.n_users, self.n_items]),
            X_indptr=self.X.indptr, X_indices=self.X.indices,
            theta_names=np.array(list(self.theta)), theta_values=np.array(list(self.theta.values())),
            beta=np.array(self.beta), use_co=np.array(self.use_collaborative),
            relations=np.array(sorted(self.incidence)),
        )
        for k, r in enumerate(sorted(self.incidence)):
            B = self.incidence[r]
            arrays[f"B{k}_indptr"], arrays[f"B{k}_indices"] = B.indptr, B.indices
            arrays[f"B{k}_ncols"] = np.array(B.shape[1])
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | FsPath) -> PathAdditiveRecommender:
        z = np.load(path, allow_pickle=False)
        n_users, n_items = (int(v) for v in z["shape"])
        X = sp.csr_matrix((np.ones(len(z["X_indices"]), np.float32), z["X_indices"], z["X_indptr"]),
                          shape=(n_users, n_items))
        inc = {}
        for k, r in enumerate(z["relations"].tolist()):
            ind = z[f"B{k}_indices"]
            inc[r] = sp.csr_matrix((np.ones(len(ind), np.float32), ind, z[f"B{k}_indptr"]),
                                   shape=(n_items, int(z[f"B{k}_ncols"])))
        theta = dict(zip(z["theta_names"].tolist(), map(float, z["theta_values"])))
        return cls(n_users, n_items, X, inc, theta, float(z["beta"]), bool(z["use_co"]))


def build_incidence(item_entity: pd.DataFrame, n_items: int, relations: list[str] | None = None
                    ) -> dict[str, sp.csr_matrix]:
    """Binary item x entity matrix per relation (entities re-indexed within each relation)."""
    out = {}
    for r, g in item_entity.groupby("relation"):
        if relations is not None and r not in relations:
            continue
        ents, codes = np.unique(g.entity.values, return_inverse=True)
        m = sp.csr_matrix((np.ones(len(g), np.float32), (g.item.values, codes)), shape=(n_items, len(ents)))
        m.data[:] = 1.0
        m.sum_duplicates()
        m.data[:] = 1.0
        out[str(r)] = m
    return out


def fit_weights(model: PathAdditiveRecommender, cfg: dict) -> tuple[dict[str, float], float, dict]:
    """Leave-one-out BPR for the non-negative path weights theta and popularity weight beta.

    For a sampled triple (u, p, n) with p in H_u and n not in H_u, both items are scored from the input history
    H_u minus p with |H_u| - 1 as normaliser. Statistics that contain the user's own interaction with p are
    corrected: co-occurrence co(i, p) - 1 and popularity pop_p - 1. The loss is the mean BPR log-loss plus an L2
    penalty, minimised with L-BFGS-B under theta, beta >= 0.
    """
    rels = model.relations
    X = model.X
    pop = model.pop
    hist = model.hist_len
    n_samples = int(cfg.get("bpr_samples", 150_000))
    reg = float(cfg.get("bpr_l2", 1e-5))
    rng = rng_for(cfg.get("seed", 0), "bpr")
    rows = np.repeat(np.arange(model.n_users), np.diff(X.indptr))
    cols = X.indices
    idx = rng.integers(0, len(cols), n_samples)
    u, p = rows[idx], cols[idx]
    n = rng.integers(0, model.n_items, n_samples)
    ok = (np.asarray(X[u, n]).ravel() == 0) & (hist[u] >= 2) & (pop[p] >= 2)
    u, p, n = u[ok], p[ok], n[ok]
    h1 = hist[u] - 1.0
    inv = 1.0 / np.sqrt(np.maximum(pop, 1.0))
    D = np.zeros((len(u), len(rels) + 1))
    mats = {r: model.relation_matrix(r).tocsc() for r in rels}
    for c in range(0, len(u), 4000):
        sl = slice(c, c + 4000)
        uc, pc, nc = u[sl], p[sl], n[sl]
        Xu = X[uc]
        for k, r in enumerate(rels):
            S = mats[r]
            sp_ = np.asarray(Xu.multiply(S[:, pc].T).sum(1)).ravel()
            sn = np.asarray(Xu.multiply(S[:, nc].T).sum(1)).ravel() - np.asarray(S[pc, nc]).ravel()
            if r == COLLAB:
                own = (np.asarray(Xu @ inv).ravel() - inv[pc]) * inv[pc]
                sp_ = (sp_ - own) * np.sqrt(pop[pc] / (pop[pc] - 1.0))
            D[sl, k] = (sp_ - sn) / h1[sl]
    D[:, -1] = np.log1p(pop[p] - 1.0) - np.log1p(pop[n])

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        x = D @ w
        loss = np.mean(np.logaddexp(0, -x)) + reg * w @ w
        g = -(D * (1.0 / (1.0 + np.exp(x)))[:, None]).mean(0) + 2 * reg * w
        return float(loss), g

    res = minimize(f, np.full(len(rels) + 1, 0.1), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * (len(rels) + 1), options={"maxiter": 500})
    theta = dict(zip(rels, map(float, res.x[:-1])))
    info = dict(triples=int(len(u)), loss=float(res.fun), converged=bool(res.success), message=str(res.message))
    log.info("path weights %s beta=%.4f (%s)", {k: round(v, 3) for k, v in theta.items()}, res.x[-1], info)
    return theta, float(res.x[-1]), info
