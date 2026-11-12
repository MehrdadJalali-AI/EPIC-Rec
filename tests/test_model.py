"""Path-additive backbone: exact path decomposition, masking without retraining, determinism, persistence."""

import numpy as np

from epic_rec.models.path_additive import PathAdditiveRecommender


def test_score_decomposes_into_paths(model):
    for u in range(0, model.n_users, 7):
        s = model.score(u)
        for j in model.topk(u, 10):
            assert abs(model.score_from_paths(u, int(j)) - s[j]) < 1e-9


def test_masking_is_row_removal_with_fixed_normaliser(model):
    u = 3
    H = model.history(u)
    M = H[:2].tolist()
    expected = model.score(u) + sum(model.masking_effect(u, i) for i in M)
    assert np.allclose(model.score(u, M)[model.candidates(u)], expected[model.candidates(u)], atol=1e-12)
    for j in model.topk(u, 5, M):
        assert abs(model.score_from_paths(u, int(j), M) - model.score(u, M)[j]) < 1e-9


def test_masking_never_raises_a_score(model):
    u = 5
    s0, s1 = model.score(u), model.score(u, model.history(u)[:3])
    c = model.candidates(u)
    assert (s1[c] <= s0[c] + 1e-12).all()


def test_paths_only_start_in_history_and_exclude_self(model):
    u = 2
    H = set(model.history(u).tolist())
    j = int(model.topk(u, 1)[0])
    for g in model.path_groups(u, j):
        assert g.origin in H and g.item == j
        if g.via_type == "user":
            assert u not in set(g.vias.tolist())


def test_disabling_a_path_lowers_score_by_its_contribution(model):
    u = 4
    j = int(model.topk(u, 1)[0])
    p = max(model.paths(u, j), key=lambda p: p.contribution)
    assert abs(model.score_from_paths(u, j) - model.score_from_paths(u, j, disabled=[p]) - p.contribution) < 1e-9


def test_topk_tie_break_and_determinism(ds, model):
    u = 1
    s = model.score(u)
    top = model.topk(u, 10)
    for a, b in zip(top[:-1], top[1:]):
        assert s[a] > s[b] or (s[a] == s[b] and a < b)
    again = PathAdditiveRecommender.fit(ds, {"seed": 0, "bpr_samples": 20000})
    assert np.array_equal(again.topk(u, 10), top)


def test_save_load_roundtrip(model, tmp_path):
    model.save(tmp_path / "m.npz")
    m2 = PathAdditiveRecommender.load(tmp_path / "m.npz")
    for u in (0, 9):
        assert np.allclose(m2.score(u), model.score(u))
