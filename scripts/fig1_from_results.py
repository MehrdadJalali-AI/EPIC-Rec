"""Redraw the Figure 1 charts from the stored report tables.

The schematic in ``manuscript/fig1_epic_rec.svg`` is unchanged. This script writes
``manuscript/fig1_charts.svg``, whose bar heights, error bars and labels are computed
from ``results/report/tables``. Both charts use the all-pairs denominator (2,000
MovieLens pairs). The conditional C1-stop rate is not drawn here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "report" / "tables"
OUT = ROOT / "manuscript" / "fig1_charts.svg"


def _row(df: pd.DataFrame, **kw) -> pd.Series:
    hit = df
    for k, v in kw.items():
        hit = hit[hit[k] == v]
    if len(hit) != 1:
        raise SystemExit(f"expected one row for {kw}, found {len(hit)}")
    return hit.iloc[0]


def load() -> dict:
    feas = pd.read_csv(TABLES / "feasibility_by_setting.csv")
    paired = pd.read_csv(TABLES / "paired_mask_size_differences.csv")
    approx = pd.read_csv(TABLES / "exact_vs_approximate.csv")
    main = dict(experiment="main", dataset="movielens-1m", K=10, alpha=0.5)
    out = {}
    for setting, key in (("A", "c1"), ("B", "c2"), ("C", "c3")):
        r = _row(feas, setting=setting, **main)
        out[key] = (float(r.mask_exists), float(r.mask_exists_lo), float(r.mask_exists_hi), int(r.pairs))
    r = _row(approx, setting="C", method="greedy:target_score:c1_stop", **main)
    out["stop"] = (float(r.success), float(r.success_lo), float(r.success_hi), int(r.pairs))
    # Conditional rate, reported in the text and not drawn on the all-pairs chart.
    out["stop_conditional"] = 100.0 * float(r.c1_c2_c3) / float(r.c1)
    for comparison, key in (("B - A", "ba"), ("C - B", "cb")):
        r = _row(paired, comparison=comparison, **main)
        out[key] = (float(r["mean"]), float(r.mean_lo), float(r.mean_hi), int(r.pairs))
        out[key + "_label"] = (round(float(r["mean"]), 3), round(float(r.mean_lo), 3), round(float(r.mean_hi), 3))
    return out


def _bar(x: float, baseline: float, scale: float, value: float, lo: float, hi: float,
         width: float, fill: str, label: str, value_text: str) -> str:
    h = value * scale
    y = baseline - h
    y_lo = baseline - lo * scale
    y_hi = baseline - hi * scale
    cx = x + width / 2
    # Degenerate intervals (a point estimate with no spread) draw no whisker.
    whisker = ""
    if hi - lo > 1e-9:
        whisker = (
            f'<line x1="{cx:.1f}" y1="{y_lo:.1f}" x2="{cx:.1f}" y2="{y_hi:.1f}" stroke="#222" stroke-width="1.6"/>'
            f'<line x1="{cx-6:.1f}" y1="{y_lo:.1f}" x2="{cx+6:.1f}" y2="{y_lo:.1f}" stroke="#222"/>'
            f'<line x1="{cx-6:.1f}" y1="{y_hi:.1f}" x2="{cx+6:.1f}" y2="{y_hi:.1f}" stroke="#222"/>'
        )
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{h:.1f}" fill="{fill}"/>'
        f"{whisker}"
        f'<text x="{cx:.1f}" y="{y_hi - 8:.1f}" font-size="16" fill="#222" text-anchor="middle" font-weight="700">{value_text}</text>'
        f'<text x="{cx:.1f}" y="{baseline + 22:.1f}" font-size="14" fill="#333" text-anchor="middle">{label}</text>'
    )


def render(data: dict) -> str:
    # Left panel: percent of all pairs. Axis 0–100 over 280 px.
    base_l, scale_l = 430.0, 2.6
    left = [
        _bar(70, base_l, scale_l, *data["c1"][:3], 52, "#d1495b", "C1", f'{data["c1"][0]:.2f}'),
        _bar(140, base_l, scale_l, *data["c2"][:3], 52, "#3b6ea5", "C1+C2", f'{data["c2"][0]:.2f}'),
        _bar(210, base_l, scale_l, *data["c3"][:3], 52, "#c9962a", "C1+C2+C3", f'{data["c3"][0]:.2f}'),
        _bar(290, base_l, scale_l, *data["stop"][:3], 62, "#9a9a9a", "C1-stop", f'{data["stop"][0]:.2f}'),
    ]
    # Right panel: paired mean size. Axis 0–2 over 280 px.
    base_r, scale_r = 430.0, 140.0
    right = [
        _bar(560, base_r, scale_r, *data["ba"][:3], 70, "#3b6ea5", "C2 added to C1", f'{data["ba_label"][0]:.3f}'),
        _bar(690, base_r, scale_r, *data["cb"][:3], 70, "#c9962a", "C3 added to C1+C2", f'{data["cb_label"][0]:.3f}'),
    ]
    ticks_l = []
    for v in (0, 25, 50, 75, 100):
        y = base_l - v * scale_l
        ticks_l.append(
            f'<line x1="48" y1="{y:.1f}" x2="54" y2="{y:.1f}" stroke="#333"/>'
            f'<text x="42" y="{y+4:.1f}" font-size="13" fill="#333" text-anchor="end">{v}</text>'
        )
    ticks_r = []
    for v in (0, 0.5, 1, 1.5, 2):
        y = base_r - v * scale_r
        ticks_r.append(
            f'<line x1="530" y1="{y:.1f}" x2="536" y2="{y:.1f}" stroke="#333"/>'
            f'<text x="524" y="{y+4:.1f}" font-size="13" fill="#333" text-anchor="end">{v:g}</text>'
        )
    n = data["c1"][3]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 500" width="860" height="500" font-family="Helvetica, Arial, sans-serif">',
        '<rect width="860" height="500" fill="#fff"/>',
        '<text x="48" y="28" font-size="18" font-weight="700" fill="#111">MovieLens-1M: valid mask, all pairs (%)</text>',
        f'<text x="48" y="50" font-size="13" fill="#444">n = {n} user–target pairs. Whiskers are 95% user-level bootstrap intervals.</text>',
        '<line x1="54" y1="150" x2="54" y2="430" stroke="#333"/>',
        '<line x1="54" y1="430" x2="380" y2="430" stroke="#333"/>',
        *ticks_l, *left,
        '<text x="530" y="28" font-size="18" font-weight="700" fill="#111">Extra interactions (mean Δ|M|)</text>',
        '<text x="530" y="50" font-size="13" fill="#444">Pairs optimal under both settings. Same intervals.</text>',
        '<line x1="536" y1="150" x2="536" y2="430" stroke="#333"/>',
        '<line x1="536" y1="430" x2="800" y2="430" stroke="#333"/>',
        *ticks_r, *right,
        '<text x="48" y="478" font-size="13" fill="#333">C1-stop is the target-score heuristic stopped at removal. Its all-pairs success is drawn here.</text>',
        "</svg>",
    ]
    return "\n".join(parts) + "\n"


def main() -> None:
    data = load()
    # Guard the denominator bug: the grey bar must be the all-pairs rate, not the conditional one.
    if abs(data["stop"][0] - 21.8) > 0.05:
        raise SystemExit(f"unexpected all-pairs C1-stop rate {data['stop'][0]}")
    if abs(data["stop_conditional"] - 27.5) > 0.15:
        raise SystemExit(f"unexpected conditional C1-stop rate {data['stop_conditional']}")
    if data["ba_label"] != (1.348, 1.185, 1.524) or data["cb_label"] != (0.313, 0.250, 0.377):
        raise SystemExit(f"paired labels {data['ba_label']} {data['cb_label']} do not match the manuscript rounding")
    text = render(data)
    for obsolete in ("2.19", "0.35", ">27.5<", ">27.50<"):
        if obsolete in text:
            raise SystemExit(f"obsolete label {obsolete} was written into the chart")
    OUT.write_text(text)
    print(f"wrote {OUT}")
    print(f"all-pairs C1-stop {data['stop'][0]:.2f} [{data['stop'][1]:.2f}, {data['stop'][2]:.2f}]")
    print(f"conditional C1-stop (not plotted) {data['stop_conditional']:.1f}")
    print(f"C2 added {data['ba'][0]:.3f} [{data['ba'][1]:.3f}, {data['ba'][2]:.3f}]")
    print(f"C3 added {data['cb'][0]:.3f} [{data['cb'][1]:.3f}, {data['cb'][2]:.3f}]")


if __name__ == "__main__":
    main()
