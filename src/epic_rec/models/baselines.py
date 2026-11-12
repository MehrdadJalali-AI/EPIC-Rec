"""Standard recommenders used to check that the path-additive backbone has reasonable accuracy.

All models expose ``score_batch(users) -> (len(users), n_items)``; evaluation excludes known interactions.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from epic_rec.utils.seeding import rng_for


class Popularity:
    def __init__(self, X: sp.csr_matrix):
        self.pop = np.asarray(X.sum(0)).ravel().astype(np.float64)

    def score_batch(self, users: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(users), 1))


class ItemKNN:
    """Item-based CF with cosine similarity, keeping the ``k`` most similar neighbours per item (k=None: all)."""

    def __init__(self, X: sp.csr_matrix, k: int | None = None):
        self.X = X.tocsr().astype(np.float64)
        pop = np.asarray(self.X.sum(0)).ravel()
        inv = 1.0 / np.sqrt(np.maximum(pop, 1.0))
        S = (sp.diags(inv) @ (self.X.T @ self.X) @ sp.diags(inv)).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        if k is not None:
            S = _top_k_rows(S, k)
        self.S = S

    def score_batch(self, users: np.ndarray) -> np.ndarray:
        return np.asarray((self.X[users] @ self.S).todense())


def _top_k_rows(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i in range(S.shape[0]):
        a, b = S.indptr[i], S.indptr[i + 1]
        if b - a > k:
            sel = np.argpartition(-S.data[a:b], k)[:k] + a
        else:
            sel = np.arange(a, b)
        rows.append(np.full(len(sel), i))
        cols.append(S.indices[sel])
        vals.append(S.data[sel])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=S.shape)


class BPRMF:
    """Matrix factorisation trained with BPR and mini-batch SGD (NumPy, CPU)."""

    def __init__(self, X: sp.csr_matrix, dim: int = 64, epochs: int = 40, lr: float = 0.05, reg: float = 1e-4,
                 batch: int = 4096, seed: int = 0):
        self.X = X.tocsr()
        self.dim, self.epochs, self.lr, self.reg, self.batch, self.seed = dim, epochs, lr, reg, batch, seed
        rng = rng_for(seed, "bprmf-init")
        self.P = rng.normal(0, 0.1, (X.shape[0], dim))
        self.Q = rng.normal(0, 0.1, (X.shape[1], dim))
        self.epochs_run = 0

    def fit(self, epochs: int | None = None, callback=None) -> BPRMF:
        """Train for ``epochs``; ``callback(model, epoch)`` may return True to stop early."""
        rows = np.repeat(np.arange(self.X.shape[0]), np.diff(self.X.indptr))
        up = np.column_stack([rows, self.X.indices])
        for ep in range(epochs or self.epochs):
            rng = rng_for(self.seed, "bprmf-epoch", self.epochs_run)
            perm = up[rng.permutation(len(up))]
            for s in range(0, len(perm), self.batch):
                b = perm[s : s + self.batch]
                u, p = b[:, 0], b[:, 1]
                n = rng.integers(0, self.X.shape[1], len(u))
                pu, qp, qn = self.P[u], self.Q[p], self.Q[n]
                x = (pu * (qp - qn)).sum(1)
                g = (1.0 / (1.0 + np.exp(x)))[:, None]
                np.add.at(self.P, u, self.lr * (g * (qp - qn) - self.reg * pu))
                np.add.at(self.Q, p, self.lr * (g * pu - self.reg * qp))
                np.add.at(self.Q, n, self.lr * (-g * pu - self.reg * qn))
            self.epochs_run += 1
            if callback is not None and callback(self, ep):
                break
        return self

    def score_batch(self, users: np.ndarray) -> np.ndarray:
        return self.P[users] @ self.Q.T
