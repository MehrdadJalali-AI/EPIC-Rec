"""Figures drawn from the result tables (never from hand-entered values). Each figure is saved as PDF and PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = {"A": "C1", "B": "C1 + C2", "C": "C1 + C2 + C3", "A3": "C1 + C3"}
COLOR = {"A": "#c8414b", "B": "#3a6ea5", "C": "#c9962a", "A3": "#7a7a7a"}
STATUS_COLOR = {"optimal": "#3a6ea5", "feasible": "#8fb3d9", "infeasible": "#c8414b", "timeout": "#bbbbbb"}


def _style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight", "figure.dpi": 150})


def _save(fig: plt.Figure, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    return paths


def _main_config(t: pd.DataFrame) -> pd.DataFrame:
    """Rows of the most frequent configuration per dataset (used when a table mixes several)."""
    if not len(t):
        return t
    keys = ["dataset", "experiment", "K", "alpha", "phi", "eps"]
    top = t.groupby(keys).pairs.sum().reset_index().sort_values("pairs").groupby("dataset").tail(1)
    return t.merge(top[keys], on=keys)


def feasibility_by_setting(t: pd.DataFrame, out: Path) -> list[Path]:
    t = _main_config(t)
    paths = []
    for d, g in t.groupby("dataset"):
        g = g.set_index("setting").reindex([s for s in LABEL if s in set(g.setting)])
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        bottom = np.zeros(len(g))
        for s in ["optimal", "feasible", "infeasible", "timeout"]:
            ax.bar([LABEL[i] for i in g.index], g[s], bottom=bottom, color=STATUS_COLOR[s], label=s)
            bottom += g[s].to_numpy()
        ax.errorbar([LABEL[i] for i in g.index], g.mask_exists,
                    yerr=[g.mask_exists - g.mask_exists_lo, g.mask_exists_hi - g.mask_exists], fmt="none",
                    ecolor="black", capsize=3, lw=0.8)
        ax.set_ylabel("pairs (%)")
        ax.set_title(f"{d}: exact-solver outcome (n = {int(g.pairs.iloc[0])} pairs)")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        paths += _save(fig, out, f"feasibility_by_setting_{d}")
    return paths


def by_stratum(t: pd.DataFrame, out: Path, col: str, xlabel: str, name: str) -> list[Path]:
    t = _main_config(t)
    paths = []
    for d, g in t.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for s, h in g.groupby("setting"):
            h = h.sort_values(col)
            ax.plot(h[col], h.mask_exists, marker="o", ms=3, color=COLOR.get(s), label=LABEL.get(s, s))
            ax.fill_between(h[col], h.mask_exists_lo, h.mask_exists_hi, color=COLOR.get(s), alpha=0.15, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mask exists (%)")
        ax.set_ylim(0, 102)
        ax.set_title(d)
        ax.legend(frameon=False, fontsize=7)
        paths += _save(fig, out, f"{name}_{d}")
    return paths


def mask_size_distribution(t: pd.DataFrame, out: Path) -> list[Path]:
    t = _main_config(t)
    paths = []
    for d, g in t.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        settings = [s for s in LABEL if s in set(g.setting)]
        w = 0.8 / max(len(settings), 1)
        for k, s in enumerate(settings):
            h = g[g.setting == s]
            ax.bar(h["size"] + (k - (len(settings) - 1) / 2) * w, 100 * h.share, width=w, color=COLOR[s],
                   label=LABEL[s])
        ax.set_xlabel("minimum mask size |M| (optimal pairs)")
        ax.set_ylabel("pairs (%)")
        ax.set_title(d)
        ax.legend(frameon=False, fontsize=7)
        paths += _save(fig, out, f"mask_size_distribution_{d}")
    return paths


def exact_vs_approx(t: pd.DataFrame, out: Path) -> list[Path]:
    t = _main_config(t)
    paths = []
    for d, g in t.groupby("dataset"):
        g = g[g.setting.isin(["B", "C"])]
        if not len(g):
            continue
        methods = sorted(g.method.unique())
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(methods) + 1.2))
        y = np.arange(len(methods))
        for k, s in enumerate(["B", "C"]):
            h = g[g.setting == s].set_index("method").reindex(methods)
            ax.barh(y + (k - 0.5) * 0.38, h.c1_c2_c3 if s == "C" else h.c1_c2, height=0.36, color=COLOR[s],
                    label=f"{LABEL[s]} satisfied")
        ax.set_yticks(y, methods, fontsize=7)
        ax.set_xlabel("pairs where the returned mask satisfies the setting (%)")
        ax.set_title(d)
        ax.legend(frameon=False, fontsize=7)
        paths += _save(fig, out, f"exact_vs_approximate_{d}")
    return paths


def evidence_survival(t: pd.DataFrame, out: Path) -> list[Path]:
    t = _main_config(t.rename(columns={"masks": "pairs"}))
    paths = []
    for d, g in t.groupby("dataset"):
        g = g[g.setting == "A"] if (g.setting == "A").any() else g
        g = g.sort_values("evidence_survival")
        fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(g) + 1.2))
        ax.barh(g.method, g.evidence_survival, color="#c9962a",
                xerr=[g.evidence_survival - g.evidence_survival_lo, g.evidence_survival_hi - g.evidence_survival])
        ax.set_xlabel("retained items keeping a faithful evidence path (%)")
        ax.set_title(f"{d}: masks returned in setting {LABEL.get(g.setting.iloc[0])}")
        paths += _save(fig, out, f"evidence_survival_{d}")
    return paths


def runtime_scalability(records: pd.DataFrame, out: Path) -> list[Path]:
    ex = records[records.method == "exact"]
    paths = []
    for d, g in ex.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for s, h in g.groupby("setting"):
            ax.scatter(h.domain_size, h.runtime, s=4, alpha=0.5, color=COLOR.get(s), label=LABEL.get(s, s))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mask domain size |D_u(t)|")
        ax.set_ylabel("exact solve time (s)")
        ax.set_title(d)
        ax.legend(frameon=False, fontsize=7, markerscale=3)
        paths += _save(fig, out, f"runtime_scalability_{d}")
    return paths


def cross_dataset(t: pd.DataFrame, out: Path) -> list[Path]:
    t = _main_config(t)
    if t.dataset.nunique() < 2:
        return []
    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    ds = sorted(t.dataset.unique())
    settings = [s for s in LABEL if s in set(t.setting)]
    w = 0.8 / len(settings)
    for k, s in enumerate(settings):
        h = t[t.setting == s].set_index("dataset").reindex(ds)
        x = np.arange(len(ds)) + (k - (len(settings) - 1) / 2) * w
        ax.bar(x, h.mask_exists, width=w, color=COLOR[s], label=LABEL[s],
               yerr=[h.mask_exists - h.mask_exists_lo, h.mask_exists_hi - h.mask_exists], capsize=2)
    ax.set_xticks(np.arange(len(ds)), ds)
    ax.set_ylabel("mask exists (%)")
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, out, "cross_dataset")


def all_figures(tables: dict[str, pd.DataFrame], records: pd.DataFrame, out: str | Path) -> list[Path]:
    _style()
    out = Path(out)
    p: list[Path] = []
    if len(tables["feasibility_by_setting"]):
        p += feasibility_by_setting(tables["feasibility_by_setting"], out)
        p += by_stratum(tables["feasibility_by_rank"], out, "rank", "target rank", "feasibility_by_rank")
        p += by_stratum(tables["feasibility_by_history_quartile"], out, "stratum",
                        "training-history quartile (0 = shortest)", "feasibility_by_history")
        p += mask_size_distribution(tables["mask_size_distribution"], out)
        p += runtime_scalability(records, out)
        p += cross_dataset(tables["cross_dataset"], out)
    if len(tables["exact_vs_approximate"]):
        p += exact_vs_approx(tables["exact_vs_approximate"], out)
    if len(tables["evidence_survival"]):
        p += evidence_survival(tables["evidence_survival"], out)
    return p
