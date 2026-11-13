"""Evidence paths: per-item evidence survival, strength and faithfulness, and readable top paths."""

from __future__ import annotations

from typing import Any

from epic_rec.models.path_additive import Path
from epic_rec.optimization.instance import ExplanationInstance


def evidence_report(inst: ExplanationInstance, M: Any, max_faithful_checks: int = 3) -> list[dict[str, Any]]:
    """Per retained item: surviving qualifying paths, their strength and faithfulness."""
    model = inst.model
    Ms = set(int(i) for i in M)
    out = []
    for k, j in enumerate(inst.retained):
        tau = float(inst.tau[k])
        rec: dict[str, Any] = dict(item=int(j), tau=tau, qualifying_origins=len(inst.Q[k]))
        if model is None:
            surv = [int(i) for i in inst.Q[k] if int(i) not in Ms]
            rec.update(surviving_paths=None, surviving_origins=len(surv), strongest=None,
                       faithful=bool(surv) and tau > 0, independent=False)
            out.append(rec)
            continue
        groups = model.path_groups(inst.u, int(j), mask=Ms)
        qual = [(g, c, x) for g in groups for x, c in zip(g.vias, g.contributions) if c >= tau - 1e-15 and c > 0]
        qual.sort(key=lambda z: -z[1])
        base = model.score_from_paths(inst.u, int(j), Ms)
        faithful = []
        for g, c, x in qual[:max_faithful_checks]:
            p = next(p for p in g.paths() if p.via == int(x))
            drop = base - model.score_from_paths(inst.u, int(j), Ms, disabled=[p])
            faithful.append(dict(origin=g.origin, relation=g.relation, via=int(x), contribution=float(c),
                                 drop=float(drop), faithful=bool(drop > 0 and drop >= inst.phi - 1e-15)))
        rec.update(
            surviving_paths=len(qual),
            surviving_origins=len({g.origin for g, _, _ in qual}),
            strongest=float(qual[0][1]) if qual else None,
            total_paths=int(sum(len(g) for g in groups)),
            faithful=any(f["faithful"] for f in faithful),
            checked=faithful,
            independent=True,
        )
        out.append(rec)
    return out


def top_paths(model: Any, u: int, j: int, mask: Any = (), n: int = 5) -> list[Path]:
    """The ``n`` strongest surviving individual paths from u's history to item j."""
    best: list[Path] = []
    for g in model.path_groups(u, int(j), mask or ()):
        order = g.contributions.argsort()[::-1][:n]
        best += [Path(g.origin, g.relation, g.via_type, int(g.vias[k]), g.item, float(g.contributions[k]))
                 for k in order]
    return sorted(best, key=lambda p: -p.contribution)[:n]
