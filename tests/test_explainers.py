"""Approximate explainers: never report unverified success, never beat the exact optimum, repair works."""

import numpy as np
import pytest

from epic_rec.evaluation.verify import verify
from epic_rec.explainers import explain, repair
from epic_rec.optimization import ExplanationInstance

METHODS = ["greedy:violation", "greedy:margin", "greedy:target_score", "greedy:list_aware", "greedy:popularity",
           "greedy:random", "beam", "continuous"]


@pytest.mark.parametrize("setting", ["A", "B", "C"])
def test_methods_against_exact(model, instances, setting):
    for u, t in instances[:12]:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        ex = explain(inst, setting, "exact")
        for m in METHODS:
            e = explain(inst, setting, m, rng=np.random.default_rng(0))
            if e.success:
                assert verify(inst, e.mask, setting)["passed"]
                assert ex.status == "optimal" and e.size >= ex.size
            assert e.status in ("found", "not_found")


def test_continuous_variants_and_repair(model, instances):
    u, t = instances[0]
    inst = ExplanationInstance(model, u, K=5)
    inst.set_target(t)
    for relax in ("sigmoid", "gumbel"):
        for init in ("attribution", "random"):
            for rep in (True, False):
                e = explain(inst, "B", "continuous", relaxation=relax, init=init, repair=rep,
                            rng=np.random.default_rng(1), steps=100)
                assert e.success == (e.mask is not None and verify(inst, e.mask, "B")["passed"])


def test_repair_restores_feasibility(model, instances):
    for u, t in instances[:10]:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        if explain(inst, "B", "exact").status != "optimal":
            continue
        r = repair(inst, [], "B")
        if r["mask"] is not None:
            assert inst.check(r["mask"], "B", 1e-6)


def test_c1_stop_is_not_counted_for_stricter_settings(model, instances):
    for u, t in instances[:10]:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        e = explain(inst, "C", "greedy:target_score:c1_stop")
        if e.mask is not None and e.success:
            assert verify(inst, e.mask, "C")["passed"]


def test_unverified_solver_output_is_rejected(model, instances, monkeypatch):
    import epic_rec.explainers.api as api

    u, t = instances[0]
    inst = ExplanationInstance(model, u, K=5)
    inst.set_target(t)
    bad = dict(status="optimal", size=0, mask=[], bound=0, gap=0.0)
    monkeypatch.setattr(api, "solve_milp", lambda *a, **k: bad)
    e = api.explain(inst, "B", "exact")
    assert e.status == "verification_failed" and not e.success
