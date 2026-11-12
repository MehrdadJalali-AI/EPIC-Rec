"""PRINCE adapter (downloads the official code; marked ``network``)."""

import pytest

from epic_rec.baselines.ppr import PPRRecommender
from epic_rec.baselines.prince import PrinceAdapter
from epic_rec.data.synthetic import make_synthetic


@pytest.mark.network
def test_prince_on_synthetic():
    ds = make_synthetic(n_users=20, n_items=30, mean_history=6, seed=1)
    rec = PPRRecommender(ds)
    r = PrinceAdapter(rec, epsilon=1e-6).explain_top1(0, K=5)
    assert r["c3"] is None
    if r["found"]:
        # PRINCE's goal is a new top-1 item; leaving the top-K list (C1) is checked separately and may fail
        assert r["prince_goal"] and r["after"][0] == r["replacement"]
        assert r["c1"] == (r["top"] not in r["after"])
