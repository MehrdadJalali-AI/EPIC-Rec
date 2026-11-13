"""Draw the schematic workflow figure used in the README (docs/figures/workflow.{png,svg}). Contains no results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
BLUE, RED, GOLD, GREY, INK = "#3a6ea5", "#c8414b", "#c9962a", "#e9edf2", "#22303c"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=GREY, ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=7.3, color=INK, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=INK))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h, y = 0.16, 0.62, 0.22
    xs = [0.012 + 0.2 * k for k in range(5)]
    box(ax, xs[0], y, w, h, "1  Data",
        "interactions +\nknowledge graph\n(MovieLens-1M, Last.fm;\nSatori linkage)\n\nchecksummed download\ndeterministic split\nleakage check", INK)
    box(ax, xs[1], y, w, h, "2  Path model",
        "s(u, j) = popularity term\n+ sum of path contributions\n\npaths  i -r- e -r- j\nand  i - user - j\n\ntop-K list L_u", BLUE)
    box(ax, xs[2], y, w, h, "3  Request",
        "remove target t from L_u\n\nmask M = set of the user's\nown interactions,\nscores recomputed\nwithout retraining", RED)
    box(ax, xs[3], y, w, h, "4  Minimum mask",
        "min |M| subject to\n\nC1  t leaves the top-K\nC2  every other item stays\nC3  every kept item keeps\n      a faithful path\n\nexact MILP or heuristics", GOLD)
    box(ax, xs[4], y, w, h, "5  Verified output",
        "independent re-scoring\nand path recomputation\n\nmask + new list + surviving\nevidence, or a certified\ninfeasibility report", INK)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w + 0.012, y + h / 2, b - 0.012, y + h / 2)
    ax.text(0.5, 0.08, "Settings:  A = C1     B = C1 + C2     C = C1 + C2 + C3     "
            "(a mask is reported as a success only after step 5 confirms every requested constraint)",
            ha="center", fontsize=8.2, color=INK)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"workflow.{ext}", dpi=200, bbox_inches="tight")
    print(OUT / "workflow.png")


if __name__ == "__main__":
    main()
