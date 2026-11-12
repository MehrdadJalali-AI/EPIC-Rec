"""User-level stratified cluster bootstrap.

Users are the resampling unit: all targets (and all configurations) of a resampled user enter together, so
several targets of one user are never treated as independent observations. Users are resampled with replacement
within strata (history-length quartiles by default), keeping each stratum's size. Ratios are estimated as
sum(numerator) / sum(denominator) over the resampled users' records, i.e. weighted by each user's number of
records.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from epic_rec.utils.seeding import rng_for


class ClusterBootstrap:
    def __init__(self, users: pd.Series, strata: pd.Series | None = None, n_boot: int = 2000, seed: int = 0):
        frame = pd.DataFrame({"user": users.values, "stratum": (strata.values if strata is not None else 0)})
        frame = frame.drop_duplicates("user")
        self.users = frame.user.to_numpy()
        self.index = {u: k for k, u in enumerate(self.users)}
        rng = rng_for(seed, "bootstrap")
        W = np.zeros((n_boot, len(self.users)))
        for _, g in frame.groupby("stratum"):
            idx = np.array([self.index[u] for u in g.user])
            draws = rng.integers(0, len(idx), (n_boot, len(idx)))
            for b in range(n_boot):
                np.add.at(W[b], idx[draws[b]], 1.0)
        self.W = W
        self.n_boot = n_boot

    def _per_user(self, df: pd.DataFrame, col: str | np.ndarray) -> np.ndarray:
        vals = df[col] if isinstance(col, str) else pd.Series(col, index=df.index)
        agg = vals.groupby(df.user.values).sum()
        out = np.zeros(len(self.users))
        for u, v in agg.items():
            if u in self.index:
                out[self.index[u]] = v
        return out

    def ratio(self, df: pd.DataFrame, num: str | np.ndarray, den: str | np.ndarray | None = None,
              level: float = 0.95) -> tuple[float, float, float]:
        """Point estimate and percentile CI of sum(num) / sum(den) (den defaults to counting records)."""
        n = self._per_user(df, num)
        d = self._per_user(df, np.ones(len(df)) if den is None else den)
        est = n.sum() / d.sum() if d.sum() > 0 else float("nan")
        bn, bd = self.W @ n, self.W @ d
        with np.errstate(invalid="ignore", divide="ignore"):
            reps = bn / bd
        reps = reps[np.isfinite(reps)]
        if not len(reps):
            return float(est), float("nan"), float("nan")
        lo, hi = np.percentile(reps, [50 * (1 - level), 100 - 50 * (1 - level)])
        return float(est), float(lo), float(hi)

    def mean(self, df: pd.DataFrame, col: str, level: float = 0.95) -> tuple[float, float, float]:
        return self.ratio(df, df[col].to_numpy(dtype=float), None, level)

    def share(self, df: pd.DataFrame, cond: Callable[[pd.DataFrame], pd.Series] | str,
              level: float = 0.95) -> tuple[float, float, float]:
        mask = cond(df) if callable(cond) else df[cond]
        return self.ratio(df, mask.astype(float).to_numpy(), None, level)
