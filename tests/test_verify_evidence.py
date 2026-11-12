from epic_rec.evaluation.verify import verify
from epic_rec.explainers import explain
from epic_rec.optimization import ExplanationInstance
from epic_rec.paths.evidence import evidence_report


def test_verification_report_contents(model, instances):
    u, t = instances[1]
    inst = ExplanationInstance(model, u, K=5)
    inst.set_target(t)
    e = explain(inst, "C", "exact")
    if not e.success:
        return
    v = verify(inst, e.mask, "C")
    assert v["independent"] and v["c1"] and v["c2"] and v["c3"]
    assert t not in v["top_k_after"] and len(v["top_k_after"]) == 5
    assert v["decomposition_error"] < 1e-9 and v["overlap"] == 1.0
    for rec in v["evidence"]:
        assert rec["faithful"]
        assert all(c["drop"] > 0 for c in rec["checked"] if c["faithful"])


def test_evidence_lost_when_its_origin_is_masked(model, instances):
    for u, t in instances:
        inst = ExplanationInstance(model, u, K=5)
        inst.set_target(t)
        for j in inst.retained:
            Q = inst.Q[int(j)] if isinstance(inst.Q, dict) else None
            if Q is not None and 0 < len(Q) <= 2:
                rep = {r["item"]: r for r in evidence_report(inst, list(Q))}
                assert not rep[int(j)]["faithful"]
                return
