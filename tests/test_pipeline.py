"""CLI smoke tests on synthetic data: demo, a tiny experiment, re-verification and the report."""

import json

import pandas as pd

from epic_rec.cli import main
from epic_rec.experiments.runner import verify_records
from epic_rec.utils.config import load_config
from epic_rec.utils.io import read_jsonl


def test_demo(tmp_path):
    main(["demo", "--out", str(tmp_path)])
    exps = json.loads((tmp_path / "explanations.json").read_text())
    assert exps and all(e["success"] == (e["verification"] is not None and e["verification"]["passed"])
                        for e in exps)


def test_run_resume_and_report(tmp_path, model):
    sets = [f"experiment.output_dir={tmp_path / 'run'}", "experiment.users_per_stratum=1", "experiment.K=[5]",
            "experiment.ranks=[1,2]", "experiment.workers=1", f"model.path={tmp_path / 'm.npz'}",
            f"results_dir={tmp_path}"]
    sets = [x for kv in sets for x in ("--set", kv)]
    cfg_path = "configs/experiments/smoke.yaml"
    main(["train", "--config", cfg_path, *sets])
    main(["run", "--config", cfg_path, *sets])
    recs = list(read_jsonl(tmp_path / "run" / "records.jsonl"))
    n = len(recs)
    assert n > 0 and not any(r["status"] == "verification_failed" for r in recs)
    main(["run", "--config", cfg_path, *sets])  # resume: nothing new
    assert len(list(read_jsonl(tmp_path / "run" / "records.jsonl"))) == n
    from epic_rec.models.path_additive import PathAdditiveRecommender

    m = PathAdditiveRecommender.load(tmp_path / "m.npz")
    assert verify_records(recs, load_config(cfg_path, sets[1::2]), m) == 0
    main(["report", str(tmp_path / "run"), "--out", str(tmp_path / "rep"), "--n-boot", "50"])
    t = pd.read_csv(tmp_path / "rep" / "tables" / "feasibility_by_setting.csv")
    assert set(t.setting) == {"A", "B", "C", "A3"}
    acc = json.loads((tmp_path / "accuracy.json").read_text())
    assert {"path_additive", "itemknn", "bprmf", "popularity"} <= set(acc["models"])
