import time

from epic_rec.optimization import run_isolated


def _hang(x):
    time.sleep(60)
    return {"status": "optimal"}


def _ok(x):
    return {"status": "optimal", "size": x}


def test_hard_kill():
    t0 = time.time()
    r = run_isolated(_hang, 1, hard_limit=1.0)
    assert r["status"] == "timeout" and r["killed"] and time.time() - t0 < 10


def test_result_passthrough():
    assert run_isolated(_ok, 3, hard_limit=10) == {"status": "optimal", "size": 3, "killed": False}
