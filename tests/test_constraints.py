"""Hand-built instances with known optima / infeasibility for every constraint."""

import numpy as np
import pytest

from epic_rec.evaluation.verify import verify
from epic_rec.optimization import EPS, ExplanationInstance, exhaustive, infeasibility_report, solve_milp
from epic_rec.optimization.milp import replacement_candidates


def make(A, s, K, P=None, t=None, alpha=0.5):
    A = np.array(A, float)
    inst = ExplanationInstance.from_arrays(A, A if P is None else np.array(P, float), np.array(s, float), K,
                                           alpha=alpha)
    inst.set_target(int(inst.L[0]) if t is None else t)
    return inst


def test_constant_score_replacement():
    # outsider 2 has no path and takes the freed slot
    inst = make([[3.0, 0, 0, 0]], [5, 4, 3, 2], K=2)
    for st in "AB":
        r = solve_milp(inst, st)
        assert r["status"] == "optimal" and r["mask"] == [0]
        assert verify(inst, r["mask"], st)["passed"]
    # retained item 1 has no path at all, so it has no evidence to preserve and C3 cannot hold
    assert inst.evidence_free == 1
    assert solve_milp(inst, "C")["status"] == "infeasible"
    assert infeasibility_report(inst, "C")["cause"].startswith("evidence-free")


def test_constant_score_replacement_with_evidence():
    inst = make([[3.0, 0, 0, 0], [0, 1.0, 0, 0]], [5, 4, 3, 2], K=2)
    r = solve_milp(inst, "C")
    assert r["status"] == "optimal" and r["mask"] == [0] and verify(inst, [0], "C")["passed"]


def test_pruning_keeps_the_only_valid_replacement():
    inst = make([[2.0, 0, 0, 0, 1.0]], [5, 4, 3, 3.5, 3.6], K=2)
    lb = inst.s - inst.A[inst.rows(inst.scope)].sum(0)
    assert 3 in replacement_candidates(inst, lb, prune=True).tolist()
    assert solve_milp(inst, "B")["size"] == 1


def test_c1_only_vs_list_preservation():
    # masking row 0 removes t but also pushes retained item 1 below outsiders 2 and 3: C1 holds, C2 cannot
    inst = make([[3.0, 2.0, 0, 0]], [5, 4.5, 3, 2.9], K=2)
    assert solve_milp(inst, "A")["size"] == 1
    assert solve_milp(inst, "B")["status"] == "infeasible"
    assert exhaustive(inst, "B", margin=EPS)["status"] == "infeasible"
    rep = infeasibility_report(inst, "B")
    assert rep["cause"].startswith("list entanglement")


def test_evidence_entanglement_known_infeasible():
    # the only interaction that lowers t is also the only qualifying evidence of retained item 1
    inst = make([[3.0, 1.0, 0, 0], [0.5, 0, 0, 0]], [5, 4.5, 3, 2], K=2)
    b = solve_milp(inst, "B")
    assert b["status"] == "optimal" and b["mask"] == [0]
    assert not verify(inst, [0], "C")["passed"]
    assert solve_milp(inst, "C")["status"] == "infeasible"
    assert exhaustive(inst, "C", margin=EPS)["status"] == "infeasible"
    assert infeasibility_report(inst, "C")["cause"].startswith("evidence entanglement")


def test_evidence_raises_the_optimum():
    # B is solved by masking row 0 alone, which deletes item 1's only strong evidence; C must use rows 1 + 2
    A = [[3.5, 1.0, 0, 0], [1.6, 0, 0, 0], [1.6, 0.2, 0, 0]]
    inst = make(A, [8, 7.5, 5, 1], K=2, alpha=0.5)
    assert solve_milp(inst, "B")["size"] == 1
    c = solve_milp(inst, "C")
    assert c["status"] == "optimal" and c["size"] == 2 and sorted(c["mask"]) == [1, 2]
    assert exhaustive(inst, "C", margin=EPS)["size"] == 2


def test_c1_unreachable():
    inst = make([[0.5, 0, 0, 0]], [5, 4, 3, 2], K=2)
    for st in "ABC":
        assert solve_milp(inst, st)["status"] == "infeasible"
    assert infeasibility_report(inst, "A")["cause"].startswith("C1 alone")


def test_only_own_history_can_be_masked():
    inst = make([[3.0, 0, 0, 0]], [5, 4, 3, 2], K=2)
    with pytest.raises(ValueError):
        verify(inst, [7], "A")


def test_settings_are_nested():
    """Every C mask satisfies B, every B mask satisfies A; optima are monotone."""
    rng = np.random.default_rng(3)
    from conftest import random_instance

    seen = 0
    while seen < 40:
        inst = random_instance(rng)
        if inst is None:
            continue
        seen += 1
        sizes = {}
        for st in "ABC":
            r = solve_milp(inst, st)
            sizes[st] = r["size"] if r["status"] == "optimal" else None
            if r["mask"] is not None:
                for weaker in "ABC"[: "ABC".index(st) + 1]:
                    assert inst.check(r["mask"], weaker, EPS)
        if sizes["C"] is not None:
            assert sizes["B"] is not None and sizes["B"] <= sizes["C"]
        if sizes["B"] is not None:
            assert sizes["A"] is not None and sizes["A"] <= sizes["B"]
