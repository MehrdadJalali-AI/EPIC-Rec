"""Personalised-PageRank recommender on the user-item-entity graph (PRINCE's native model).

Nodes: users, items and (relation, entity) nodes. Edges: user <-> item for training interactions, item <-> entity
for KG links, and a self-loop on every node. Transition weights are uniform over a node's out-edges. The score of
item j for user u is the PageRank of j personalised on u with restart probability ``alpha``. Masking removes the
user's out-edges to the masked items and renormalises the remaining out-edges (as in PRINCE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from epic_rec.data.dataset import InteractionDataset


class PPRRecommender:
    def __init__(self, ds: InteractionDataset, alpha: float = 0.15, tol: float = 1e-10, max_iter: int = 1000):
        self.ds, self.alpha, self.tol, self.max_iter = ds, alpha, tol, max_iter
        self.n_users, self.n_items = ds.n_users, ds.n_items
        ie = ds.item_entity
        ent_keys = sorted(set(zip(ie.relation, ie.entity)))
        self.ent_index = {k: m for m, k in enumerate(ent_keys)}
        self.n_nodes = self.n_users + self.n_items + len(ent_keys)
        rows, cols = [], []
        tr = ds.train
        rows += list(tr[:, 0]) + list(self.n_users + tr[:, 1])
        cols += list(self.n_users + tr[:, 1]) + list(tr[:, 0])
        e = np.array([self.n_users + self.n_items + self.ent_index[(r, x)] for r, x in zip(ie.relation, ie.entity)],
                     dtype=np.int64)
        it = self.n_users + ie.item.to_numpy()
        rows += list(it) + list(e)
        cols += list(e) + list(it)
        rows += list(range(self.n_nodes))
        cols += list(range(self.n_nodes))
        A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n_nodes, self.n_nodes))
        A.data[:] = 1.0
        self.A = A

    def item_node(self, j: int) -> int:
        return self.n_users + int(j)

    def history(self, u: int) -> np.ndarray:
        row = self.A[u].indices
        return np.sort(row[(row >= self.n_users) & (row < self.n_users + self.n_items)] - self.n_users)

    def transition(self, u: int | None = None, mask: Any = ()) -> sp.csr_matrix:
        A = self.A
        if u is not None and len(mask):
            A = A.tolil(copy=True)
            for i in mask:
                A[u, self.item_node(int(i))] = 0
            A = A.tocsr()
            A.eliminate_zeros()
        deg = np.asarray(A.sum(1)).ravel()
        return (sp.diags(1.0 / deg) @ A).tocsr()

    def scores(self, u: int, mask: Any = ()) -> np.ndarray:
        """PPR of every item personalised on u (power iteration, L1 tolerance ``tol``)."""
        T = self.transition(u, mask).T.tocsr()
        e = np.zeros(self.n_nodes)
        e[u] = 1.0
        x = e.copy()
        for _ in range(self.max_iter):
            nx = self.alpha * e + (1 - self.alpha) * (T @ x)
            if np.abs(nx - x).sum() < self.tol:
                x = nx
                break
            x = nx
        return x[self.n_users : self.n_users + self.n_items]

    def topk(self, u: int, K: int, mask: Any = ()) -> np.ndarray:
        s = self.scores(u, mask)
        idx = np.flatnonzero(~np.isin(np.arange(self.n_items), self.history(u)))
        return idx[np.lexsort((idx, -s[idx]))[:K]]

    def networkx_graph(self) -> Any:
        """Weighted DiGraph in PRINCE's format (edge attribute ``weight`` = transition probability)."""
        import networkx as nx

        T = self.transition()
        g = nx.DiGraph()
        g.add_nodes_from(range(self.n_nodes))
        coo = T.tocoo()
        g.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))
        return g
