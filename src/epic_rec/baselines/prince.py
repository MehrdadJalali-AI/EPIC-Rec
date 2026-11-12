"""Adapter for the official PRINCE implementation (Ghazimatin et al., WSDM 2020).

The official code (github.com/azinmatin/prince, pinned commit) is downloaded into a local cache and its SHA-256
checked. It is Python 2; the only local change is converting ``print message`` to ``print(message)`` so that it
imports under Python 3. No PRINCE code is stored in this repository.

PRINCE explains the top-1 recommendation of a personalised-PageRank recommender: it returns a smallest set of
the user's actions whose removal makes another item outrank the top item. The adapter

1. builds PRINCE's native recommender (:class:`epic_rec.baselines.ppr.PPRRecommender`) and its networkx graph;
2. computes reverse-push PPR estimates for the user's top-K items with the official ``reverse_local_push``;
3. calls the official ``CFE.compute_pagerank_wo_u`` and ``CFE.cfe_item_centric_algo_poly`` with the top-K items
   as replacement candidates;
4. evaluates the returned deletion set with exact power-iteration PageRank: PRINCE's own goal (top item no longer
   rank 1), C1 (top item out of the top-K) and C2 (other top-K items stay). C3 is not applicable to PageRank.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from epic_rec.baselines.ppr import PPRRecommender
from epic_rec.baselines.registry import PUBLISHED
from epic_rec.utils.io import sha256_file

FILES = {
    "cfe_generator.py": "af2a1b18be27079f4ff360c80f49b5089bfec107213618a32b8b725e44bb7a2c",
    "local_push.py": "f70c2ed16fa4b804dc4f16e246fe138de5ff990074460eac05941c4fc8f59b40",
}


def fetch_prince(cache: str | Path = "~/.cache/epic_rec/prince") -> Path:
    commit = PUBLISHED["prince"].commit
    d = Path(cache).expanduser() / commit
    d.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        orig = d / f"orig_{name}"
        if not orig.exists():
            url = f"https://raw.githubusercontent.com/azinmatin/prince/{commit}/{name}"
            with urllib.request.urlopen(url, timeout=60) as r:
                orig.write_bytes(r.read())
        if sha256_file(orig) != digest:
            raise RuntimeError(f"{orig} does not match the pinned PRINCE file ({digest})")
        src = orig.read_text()
        src = re.sub(r"^(\s*)print (?!\()(.*)$", r"\1print(\2)", src, flags=re.M)
        (d / name).write_text(src)
    return d


def _load(d: Path) -> tuple[Any, Any]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("local_push", d / "local_push.py")
    lp = importlib.util.module_from_spec(spec)
    sys.modules["local_push"] = lp
    spec.loader.exec_module(lp)
    spec = importlib.util.spec_from_file_location("cfe_generator", d / "cfe_generator.py")
    cg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cg)
    return lp, cg


class PrinceAdapter:
    def __init__(self, rec: PPRRecommender, epsilon: float | None = None,
                 cache: str | Path = "~/.cache/epic_rec/prince"):
        self.rec = rec
        self.lp, self.cg = _load(fetch_prince(cache))
        self.graph = rec.networkx_graph()
        self.epsilon = epsilon if epsilon is not None else 1.0 / (rec.n_nodes * 10_000)

    def explain_top1(self, u: int, K: int = 10) -> dict[str, Any]:
        t0 = time.time()
        rec = self.rec
        L = [int(j) for j in rec.topk(u, K)]
        nodes = [rec.item_node(j) for j in L]
        p, r = {}, {}
        for n in nodes:
            p[n], r[n] = self.lp.reverse_local_push(self.graph, n, {}, {}, alpha=rec.alpha, e=self.epsilon)
        cfe = self.cg.CFE(self.graph, p, r, rec.alpha, self.epsilon)
        cfe.compute_pagerank_wo_u(u, nodes)
        actions, replacement, size = cfe.cfe_item_centric_algo_poly(u, nodes[0], nodes)
        mask = sorted(int(a) - rec.n_users for a in actions if rec.n_users <= int(a) < rec.n_users + rec.n_items)
        other = [int(a) for a in actions if not rec.n_users <= int(a) < rec.n_users + rec.n_items]
        out: dict[str, Any] = dict(user=int(u), K=K, top=L[0], list=L, mask=mask, size=len(mask),
                                   non_item_actions=other, found=bool(actions),
                                   replacement=(int(replacement) - rec.n_users) if replacement is not None else None,
                                   epsilon=self.epsilon, runtime=None)
        if actions:
            after = [int(j) for j in rec.topk(u, K, mask)]
            out.update(after=after, prince_goal=after[0] != L[0], c1=L[0] not in after,
                       c2=all(j in after for j in L[1:]), c3=None)
        out["runtime"] = time.time() - t0
        return out
