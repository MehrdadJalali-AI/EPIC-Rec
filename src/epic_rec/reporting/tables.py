"""Machine-readable result tables computed from experiment records.

Every number in every table is computed from ``records.jsonl`` files; nothing is entered by hand. Confidence
intervals are 95% user-level stratified cluster bootstrap intervals (:class:`ClusterBootstrap`), and every table
states its denominators (``pairs``, ``users``). A *pair* is one (user, target) at one configuration
(experiment, K, alpha, phi, eps).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epic_rec.evaluation.bootstrap import ClusterBootstrap
from epic_rec.utils.io import read_jsonl

CONFIG = ["dataset", "experiment", "K", "alpha", "phi", "eps"]
PAIR = CONFIG + ["user", "rank", "target"]
EXACT = "exact"
STATUSES = ["optimal", "feasible", "infeasible", "timeout"]


def load_records(paths: Iterable[str | Path]) -> pd.DataFrame:
    rows = [r for p in paths for r in read_jsonl(p)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["exists"] = df.status.isin(["optimal", "feasible", "found"]) & df.success.astype(bool)
    return df


class Bootstraps:
    """One ClusterBootstrap per dataset, stratified by history quartile."""

    def __init__(self, df: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
        self.b = {}
        for d, g in df.groupby("dataset"):
            users = g.drop_duplicates("user")
            self.b[d] = ClusterBootstrap(users.user, users.stratum, n_boot=n_boot, seed=seed)

    def share(self, g: pd.DataFrame, mask: pd.Series) -> tuple[float, float, float]:
        if not len(g):
            return (np.nan, np.nan, np.nan)
        return self.b[g.dataset.iloc[0]].ratio(g, mask.astype(float).to_numpy())

    def mean(self, g: pd.DataFrame, values: np.ndarray) -> tuple[float, float, float]:
        if not len(g):
            return (np.nan, np.nan, np.nan)
        return self.b[g.dataset.iloc[0]].ratio(g, np.asarray(values, dtype=float))


def _ci(prefix: str, est: tuple[float, float, float], scale: float = 100.0) -> dict[str, float]:
    return {prefix: est[0] * scale, f"{prefix}_lo": est[1] * scale, f"{prefix}_hi": est[2] * scale}


def feasibility(df: pd.DataFrame, bs: Bootstraps, extra: list[str] | None = None) -> pd.DataFrame:
    """Exact-solver outcomes per configuration and setting (percent of pairs)."""
    ex = df[df.method == EXACT]
    out = []
    for key, g in ex.groupby(CONFIG + (extra or []) + ["setting"], dropna=False):
        row: dict[str, Any] = dict(zip(CONFIG + (extra or []) + ["setting"], key))
        row.update(setting_label=g.setting_label.iloc[0], pairs=len(g), users=g.user.nunique())
        for s in STATUSES:
            row[s] = 100 * (g.status == s).mean()
        row["verification_failed"] = int((g.status == "verification_failed").sum())
        row.update(_ci("mask_exists", bs.share(g, g.exists)))
        row["killed"] = int(g.killed.sum())
        row["runtime_median_s"] = g.runtime.median()
        row["runtime_max_s"] = g.runtime.max()
        feas = g[g.status == "feasible"]
        row["gap_mean_feasible"] = feas.gap.mean() if len(feas) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def mask_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    ex = df[(df.method == EXACT) & (df.status == "optimal")]
    t = ex.groupby(CONFIG + ["setting", "size"]).size().rename("pairs").reset_index()
    t["share"] = t.pairs / t.groupby(CONFIG + ["setting"]).pairs.transform("sum")
    return t


def paired_differences(df: pd.DataFrame, bs: Bootstraps,
                       pairs: tuple[tuple[str, str], ...] = (("A", "B"), ("B", "C"), ("A", "C"), ("A", "A3"),
                                                             ("A3", "C"))) -> pd.DataFrame:
    """Increase in minimum mask size between settings, on the pairs optimal in both settings."""
    ex = df[(df.method == EXACT) & (df.status == "optimal")]
    wide = ex.pivot_table(index=PAIR + ["stratum"], columns="setting", values="size", aggfunc="first").reset_index()
    out = []
    for key, g in wide.groupby(CONFIG):
        for a, b in pairs:
            if a not in g or b not in g:
                continue
            h = g.dropna(subset=[a, b])
            if not len(h):
                continue
            d = (h[b] - h[a]).to_numpy()
            row = dict(zip(CONFIG, key), comparison=f"{b} - {a}", pairs=len(h), users=h.user.nunique(),
                       pairs_total=len(g))
            row.update(_ci("mean", bs.mean(h, d), 1.0))
            row.update(median=float(np.median(d)), share_positive=100 * (d > 0).mean(),
                       share_negative=100 * (d < 0).mean())
            out.append(row)
    return pd.DataFrame(out)


def stratified(df: pd.DataFrame, bs: Bootstraps, by: str) -> pd.DataFrame:
    """Mask existence by a stratum column (rank, stratum, entanglement quartile, ...)."""
    ex = df[df.method == EXACT]
    out = []
    for key, g in ex.groupby(CONFIG + ["setting", by], dropna=False):
        row = dict(zip(CONFIG + ["setting", by], key), pairs=len(g), users=g.user.nunique())
        row.update(_ci("mask_exists", bs.share(g, g.exists)))
        row.update(infeasible=100 * (g.status == "infeasible").mean(), timeout=100 * (g.status == "timeout").mean())
        out.append(row)
    return pd.DataFrame(out)


def add_entanglement_quartile(df: pd.DataFrame, col: str = "ent_shared_evidence_frac") -> pd.DataFrame:
    df = df.copy()
    if col in df:
        firsts = df.drop_duplicates(PAIR)
        cuts = np.unique(np.quantile(firsts[col].dropna(), [0.25, 0.5, 0.75]))
        df["entanglement_quartile"] = np.digitize(df[col].fillna(0), cuts, right=True)
    return df


def exact_vs_approx(df: pd.DataFrame, bs: Bootstraps) -> pd.DataFrame:
    """Every non-exact method against the exact solver on the same pairs and setting."""
    ex = df[df.method == EXACT][PAIR + ["setting", "status", "size"]].rename(
        columns={"status": "exact_status", "size": "opt"})
    ap = df[df.method != EXACT].merge(ex, on=PAIR + ["setting"], how="left")
    out = []
    for key, g in ap.groupby(CONFIG + ["setting", "method"]):
        row = dict(zip(CONFIG + ["setting", "method"], key), pairs=len(g), users=g.user.nunique())
        c1 = g.c1.fillna(False).astype(bool) & g.c1_margin.fillna(False).astype(bool)
        c12 = c1 & g.c2.fillna(False).astype(bool) & g.c1_c2_margin.fillna(False).astype(bool)
        c123 = c12 & g.c3.fillna(False).astype(bool)
        row.update(_ci("success", bs.share(g, g.success.astype(bool))))
        row.update(_ci("c1", bs.share(g, c1)))
        row.update(_ci("c1_c2", bs.share(g, c12)))
        row.update(_ci("c1_c2_c3", bs.share(g, c123)))
        opt = g[g.exact_status == "optimal"]
        row["pairs_exact_optimal"] = len(opt)
        row["success_on_optimal"] = 100 * opt.success.mean() if len(opt) else np.nan
        hit = opt[opt.success.astype(bool)]
        row["exact_size_share"] = 100 * (hit["size"] == hit.opt).mean() if len(hit) else np.nan
        row["excess_size_mean"] = (hit["size"] - hit.opt).mean() if len(hit) else np.nan
        row["smaller_than_optimum"] = int((hit["size"] < hit.opt).sum())
        inf = g[g.exact_status == "infeasible"]
        row["pairs_exact_infeasible"] = len(inf)
        row["success_on_infeasible"] = int(inf.success.astype(bool).sum())
        row["runtime_median_s"] = g.runtime.median()
        out.append(row)
    return pd.DataFrame(out)


def evidence(df: pd.DataFrame, bs: Bootstraps) -> pd.DataFrame:
    """Evidence survival and path strength of the masks each method returned."""
    g0 = df[df["mask"].notna() & df.evidence_survival.notna()]
    out = []
    for key, g in g0.groupby(CONFIG + ["setting", "method"]):
        row = dict(zip(CONFIG + ["setting", "method"], key), masks=len(g), users=g.user.nunique())
        row.update(_ci("evidence_survival", bs.mean(g, g.evidence_survival.to_numpy())))
        row["surviving_paths_mean"] = g.surviving_paths_mean.mean()
        row["strongest_min_mean"] = g.strongest_min.mean()
        row["overlap_mean"] = g.overlap.mean()
        row["kendall_mean"] = g.kendall.mean()
        out.append(row)
    return pd.DataFrame(out)


def runtime(df: pd.DataFrame) -> pd.DataFrame:
    ex = df[df.method == EXACT].copy()
    ex["domain_bin"] = pd.cut(ex.domain_size, [0, 10, 20, 50, 100, 200, 500, 10**6], right=True)
    out = []
    for key, g in ex.groupby(CONFIG + ["setting", "domain_bin"], observed=True):
        out.append(dict(zip(CONFIG + ["setting", "domain_bin"], map(str, key)), pairs=len(g),
                        runtime_median_s=g.runtime.median(), runtime_p90_s=g.runtime.quantile(0.9),
                        runtime_max_s=g.runtime.max(), timeout=100 * (g.status == "timeout").mean(),
                        killed=int(g.killed.sum())))
    return pd.DataFrame(out)


def semantics(df: pd.DataFrame) -> pd.DataFrame:
    """P versus P_eps on pairs solved by enumeration, and MILP agreement with the P_eps enumeration."""
    if "enum_P" not in df:
        return pd.DataFrame()
    ex = df[(df.method == EXACT) & df.enum_P.notna()]
    out = []
    for key, g in ex.groupby(CONFIG + ["setting"]):
        P, E = g.enum_P.astype(int), g.enum_Peps.astype(int)
        decided = g.status.isin(["optimal", "infeasible"])
        milp = np.where(g.status == "infeasible", -1, g["size"].fillna(-2)).astype(int)
        out.append(dict(zip(CONFIG + ["setting"], key), pairs=len(g), users=g.user.nunique(),
                        status_differs=int(((P < 0) != (E < 0)).sum()),
                        feasible_only_in_P=int(((P >= 0) & (E < 0)).sum()),
                        optimum_differs=int(((P >= 0) & (E >= 0) & (P != E)).sum()),
                        milp_decided=int(decided.sum()), milp_agrees=int(((milp == E.to_numpy()) & decided).sum())))
    return pd.DataFrame(out)


def infeasibility_causes(df: pd.DataFrame) -> pd.DataFrame:
    if "infeasibility_cause" not in df:
        return pd.DataFrame()
    ex = df[(df.method == EXACT) & (df.status == "infeasible")]
    t = ex.groupby(CONFIG + ["setting", "infeasibility_cause"]).size().rename("pairs").reset_index()
    t["share"] = 100 * t.pairs / t.groupby(CONFIG + ["setting"]).pairs.transform("sum")
    return t


def all_tables(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    df = add_entanglement_quartile(df)
    bs = Bootstraps(df, n_boot=n_boot, seed=seed)
    return {
        "feasibility_by_setting": feasibility(df, bs),
        "mask_size_distribution": mask_size_distribution(df),
        "paired_mask_size_differences": paired_differences(df, bs),
        "feasibility_by_rank": stratified(df, bs, "rank"),
        "feasibility_by_history_quartile": stratified(df, bs, "stratum"),
        "feasibility_by_entanglement_quartile": stratified(df, bs, "entanglement_quartile")
        if "entanglement_quartile" in df else pd.DataFrame(),
        "exact_vs_approximate": exact_vs_approx(df, bs),
        "evidence_survival": evidence(df, bs),
        "runtime_scalability": runtime(df),
        "p_vs_p_eps": semantics(df),
        "infeasibility_causes": infeasibility_causes(df),
        "cross_dataset": feasibility(df, bs)[CONFIG + ["setting", "pairs", "users", "mask_exists", "mask_exists_lo",
                                                       "mask_exists_hi", "infeasible", "timeout"]],
    }


def write_tables(tables: dict[str, pd.DataFrame], out: str | Path) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, t in tables.items():
        p = out / f"{name}.csv"
        t.to_csv(p, index=False, float_format="%.6g")
        paths.append(p)
    return paths
