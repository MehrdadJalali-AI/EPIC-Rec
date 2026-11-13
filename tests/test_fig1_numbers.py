"""The Figure 1 charts must be drawn from the stored tables, on one denominator."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "report" / "tables"


@pytest.mark.skipif(not (TABLES / "paired_mask_size_differences.csv").exists(), reason="report tables are not in this checkout")
def test_figure1_charts_match_stored_tables():
    from scripts.fig1_from_results import load

    data = load()
    assert data["c1"][3] == data["stop"][3] == 2000
    assert data["c3"][0] == pytest.approx(73.10, abs=0.01)
    assert data["stop"][0] == pytest.approx(21.8, abs=0.05)
    # The conditional rate is a different denominator and must not be used as the bar.
    assert data["stop_conditional"] == pytest.approx(27.5, abs=0.15)
    assert data["stop"][0] != pytest.approx(data["stop_conditional"], abs=1)
    assert data["ba_label"] == (1.348, 1.185, 1.524)
    assert data["cb_label"] == (0.313, 0.250, 0.377)
    svg = (ROOT / "manuscript" / "fig1_charts.svg").read_text()
    for label in ("73.10", "21.80", "1.348", "0.313"):
        assert label in svg
    for obsolete in ("2.19", ">27.5<", "0.35"):
        assert obsolete not in svg
