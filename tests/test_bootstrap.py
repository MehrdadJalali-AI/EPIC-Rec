import numpy as np
import pandas as pd

from epic_rec.evaluation.bootstrap import ClusterBootstrap


def frame():
    rng = np.random.default_rng(0)
    rows = [dict(user=u, stratum=u % 4, y=float(rng.random() < 0.3 + 0.1 * (u % 4)))
            for u in range(80) for _ in range(rng.integers(1, 6))]
    return pd.DataFrame(rows)


def test_ci_contains_estimate_and_is_deterministic():
    df = frame()
    b1 = ClusterBootstrap(df.user, df.stratum, n_boot=500, seed=1)
    b2 = ClusterBootstrap(df.user, df.stratum, n_boot=500, seed=1)
    e = b1.share(df, "y")
    assert e == b2.share(df, "y")
    assert e[1] <= e[0] <= e[2]
    assert abs(e[0] - df.y.mean()) < 1e-12


def test_resampling_keeps_stratum_sizes():
    df = frame()
    b = ClusterBootstrap(df.user, df.stratum, n_boot=50, seed=0)
    users = df.drop_duplicates("user")
    for _q, g in users.groupby("stratum"):
        idx = [b.index[u] for u in g.user]
        assert np.allclose(b.W[:, idx].sum(1), len(g))
