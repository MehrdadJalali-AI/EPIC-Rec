"""The MILP's optimum equals exhaustive enumeration under P_eps, with and without replacement pruning."""

import numpy as np
import pytest

from conftest import random_instance
from epic_rec.evaluation.verify import verify
from epic_rec.optimization import EPS, ExplanationInstance, exhaustive, solve_milp


@pytest.mark.parametrize("setting", ["A", "B", "C", "A3"])
def test_random_instances(setting):
    rng = np.random.default_rng({"A": 0, "B": 1, "C": 2, "A3": 3}[setting])
    n = 0
    while n < 60:
        inst = random_instance(rng)
        if inst is None:
            continue
        n += 1
        e = exhaustive(inst, setting, 14, margin=EPS)
        for prune in (True, False):
            m = solve_milp(inst, setting, prune=prune, time_limit=30)
            assert m["status"] in ("optimal", "infeasible")
            if e["status"] == "infeasible":
                assert m["status"] == "infeasible"
            else:
                assert m["status"] == "optimal" and m["size"] == e["size"]
                assert verify(inst, m["mask"], setting)["passed"]


def test_model_instances(model, instances):
    for u, t in instances:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        for st in ("A", "B", "C"):
            e = exhaustive(inst, st, 12, margin=EPS)
            m = solve_milp(inst, st)
            assert (m["status"] == "infeasible") == (e["status"] == "infeasible")
            if m["status"] == "optimal":
                assert m["size"] == e["size"]
                v = verify(inst, m["mask"], st)
                assert v["passed"] and v["independent"]
