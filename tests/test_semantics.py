"""Regression tests for exact ranking (P, id tie-break) versus the margin problem P_eps."""

import numpy as np

from epic_rec.evaluation.verify import verify
from epic_rec.explainers.api import explain
from epic_rec.optimization import EPS, ExplanationInstance, exhaustive, solve_milp


def tie_instance():
    # after masking row 0, t (id 2) and outsider 0 both score 3: P ranks item 0 first (lower id), P_eps needs a gap
    inst = ExplanationInstance.from_arrays(np.array([[0.0, 0.0, 2.0]]), np.array([[0.0, 0.0, 2.0]]),
                                           np.array([3.0, 4.0, 5.0]), 2)
    inst.set_target(2)
    return inst


def test_tie_feasible_under_P_only():
    inst = tie_instance()
    assert exhaustive(inst, "B", 14, margin=0.0)["size"] == 1
    assert exhaustive(inst, "B", 14, margin=EPS)["status"] == "infeasible"
    assert solve_milp(inst, "B")["status"] == "infeasible"


def test_verifier_requires_both_semantics():
    v = verify(tie_instance(), [0], "B")
    assert v["passed_P"] and not v["passed_P_eps"] and not v["passed"]


def test_heuristics_do_not_report_P_only_masks():
    inst = tie_instance()
    for m in ("greedy:violation", "greedy:target_score", "beam", "continuous"):
        e = explain(inst, "B", m, rng=np.random.default_rng(0))
        assert not e.success


def test_margin_monotone():
    """A larger eps can only make the problem harder."""
    inst = ExplanationInstance.from_arrays(np.array([[1.0, 0, 0.0, 0], [0.5, 0, 0.2, 0]]),
                                           np.array([[1.0, 0, 0.0, 0], [0.5, 0, 0.2, 0]]),
                                           np.array([4.0, 3.9, 3.2, 1.0]), 2)
    inst.set_target(0)
    sizes = [solve_milp(inst, "B", eps=e)["size"] for e in (1e-8, 1e-6, 1e-2, 0.6)]
    found = [s for s in sizes if s is not None]
    assert found == sorted(found)
    assert sizes[-1] is None
