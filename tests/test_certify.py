"""Certification confirms true MILP claims and corrects wrong ones to the enumerated optimum."""

from epic_rec.evaluation.verify import verify
from epic_rec.optimization import EPS, ExplanationInstance, exhaustive, solve_milp
from epic_rec.optimization.certify import certify, certify_isolated


def test_certify_confirms_and_corrects(model, instances):
    seen = set()
    for u, t in instances:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        for st in ("A", "B", "C"):
            e = exhaustive(inst, st, 12, margin=EPS)
            m = solve_milp(inst, st)
            c = certify(inst, st, m)
            if m["status"] == "infeasible" and not m["iterations"]:
                assert c["certification"] == "structural"
                continue
            if m["status"] == "optimal" and m["size"] == 0:
                assert c["certification"] == "structural"
                continue
            assert c["certification"] == "confirmed" and c["status"] == m["status"] and c["size"] == m["size"]
            seen.add(m["status"])
            if e["status"] == "optimal":
                # a claimed optimum one too large, and a wrong infeasibility claim, are both corrected
                for wrong in (dict(m, size=m["size"] + 1), dict(m, status="infeasible", size=None, mask=None)):
                    c = certify_isolated(inst, st, wrong)
                    assert c["certification"] == "corrected" and c["status"] == "optimal"
                    assert c["size"] == e["size"] and c["original_status"] == wrong["status"]
                    assert verify(inst, c["mask"], st)["passed"]
    assert seen == {"optimal", "infeasible"}


def test_certify_skips_other_statuses(model, instances):
    u, t = instances[0]
    inst = ExplanationInstance(model, u, K=5)
    inst.set_target(t)
    raw = dict(status="timeout", size=None, mask=None, bound=None, gap=None, iterations=3)
    assert certify(inst, "C", raw)["certification"] == "not_applicable"
