import numpy as np
import pandas as pd
import pytest

from epic_rec.data import InteractionDataset, split_per_user
from epic_rec.data.preprocess import item_entity_links, k_core
from epic_rec.utils.config import load_config


def test_split_deterministic_and_disjoint():
    rng = np.random.default_rng(0)
    pairs = np.unique(np.column_stack([rng.integers(0, 30, 600), rng.integers(0, 50, 600)]), axis=0)
    a = split_per_user(pairs, 30, 0.1, 0.2, seed=5)
    b = split_per_user(pairs, 30, 0.1, 0.2, seed=5)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    sets = [set(map(tuple, s.tolist())) for s in a]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sum(map(len, sets)) == len(pairs)
    tr_users = set(a[0][:, 0].tolist())
    assert set(a[2][:, 0].tolist()) <= tr_users


def test_leakage_check_detects_overlap(ds):
    ds.check_no_leakage()
    bad = InteractionDataset(ds.name, ds.n_users, ds.n_items, ds.train, ds.val, np.vstack([ds.test, ds.train[:1]]),
                             ds.item_entity, ds.user_ids, ds.item_ids, ds.meta)
    with pytest.raises(AssertionError):
        bad.check_no_leakage()


def test_k_core_converges():
    r = pd.DataFrame({"user": [0, 0, 0, 1, 1, 2], "item": [0, 1, 2, 0, 1, 0]})
    out, _ = k_core(r, min_user=2, min_item=2)
    assert (out.user.value_counts() >= 2).all() and (out.item.value_counts() >= 2).all()


def test_item_entity_links_filters():
    kg = pd.DataFrame({"h": [100, 101, 102, 100, 100], "r": ["g", "g", "g", "d", "g"], "t": [7, 7, 8, 9, 101]})
    ie, counts = item_entity_links(kg, {0: 100, 1: 101, 2: 102}, np.array([0, 1, 2]), {"min_items_per_entity": 2})
    assert set(map(tuple, ie[["item_orig", "relation", "entity"]].values.tolist())) == {(0, "g", 7), (1, "g", 7)}
    assert counts["item_item_triples_ignored"] == 1


def test_save_load_roundtrip(ds, tmp_path):
    ds.save(tmp_path / "d")
    d2 = InteractionDataset.load(tmp_path / "d")
    assert np.array_equal(d2.train, ds.train) and d2.stats() == ds.stats()


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nexperiment: {K: [10], time_limit: 20}\n")
    (tmp_path / "x.yaml").write_text("base: base.yaml\nexperiment: {K: [5]}\n")
    cfg = load_config(tmp_path / "x.yaml", ["experiment.time_limit=3"])
    assert cfg["a"] == 1 and cfg["experiment"] == {"K": [5], "time_limit": 3}
