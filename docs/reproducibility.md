# Reproducibility

## Environment

* Python ≥ 3.10 (developed and tested with 3.12; CI also runs 3.10).
* Exact versions are pinned in `requirements.txt`; `pyproject.toml` pins only `highspy==1.15.1`, because
  `highspy` 1.16.0 was observed to hang on some models of this project. Each exact solve also runs in a child
  process with a hard kill (`experiment.hard_kill_grace`), so a hang cannot stall a run; killed solves are recorded
  as `timeout` with `killed = true` and counted in every table.
* CPU only. No GPU code.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[dev,prince]"
```

## Determinism

* Every random choice draws from a generator derived from a fixed key, `rng_for(seed, "purpose", ...)` (SHA-256 of
  the key), never from global state: data splits (per user), the LOO-BPR sample, user sampling (per history
  stratum), random-order baselines and random initialisations (per user, target, setting and method), and bootstrap
  resampling. Results therefore do not depend on the number of worker processes or on job order.
* Rankings break ties by item id; the MILP uses `threads = 1`, `mip_rel_gap = 0` and an integral objective.
* Scores, matrices and path contributions are float64. The path decomposition is checked to `1e-9`.
* Exact solves can still differ between machines when a time limit is hit (`feasible` / `timeout` outcomes
  depend on speed). All tables report these outcomes explicitly.

## Provenance and resumability

* `data/raw/<dataset>/download_manifest.json`: URL, SHA-256 and size of every raw file.
* `data/processed/<dataset>/manifest.json`: raw-file checksums, every preprocessing setting, filtering counts,
  dataset statistics, and SHA-256 of the processed arrays.
* `results/<dataset>/accuracy.json`: dataset statistics, learned path weights, selected baseline
  hyper-parameters and validation / test accuracy.
* `results/<dataset>/<experiment>/sample.json`: the sampled users, the history-quartile cut points and the full
  resolved config.
* `records.jsonl`: one line per (pair, setting, method), appended with `fsync` as jobs finish. Restarting the same
  command skips finished jobs, so long runs can be interrupted and resumed.
* `results/report/report_manifest.json`: the record files and the bootstrap settings used for every table and
  figure.

## Commands

| goal | command | time (indicative, 4 CPU workers) |
| --- | --- | --- |
| tests | `pytest -q -m "not network"` | under a minute |
| demo | `epic-rec demo` | seconds |
| smoke experiment + report | `bash scripts/run_smoke.sh` | minutes |
| full study | `bash scripts/run_full.sh` | several hours |

Configs are YAML with inheritance (`base:`); any key can be overridden on the command line, e.g.
`epic-rec run --config configs/experiments/movielens_main.yaml --set experiment.time_limit=5 --set experiment.workers=8`.
`EPIC_WORKERS` overrides the number of worker processes.

## What is not reproduced bit-for-bit

The manuscript draft's experiments were run with an earlier research prototype that used an 80 / 20 split
without a validation set. This implementation follows the same definitions (path-additive model, P_ε, C1, C2,
C3 with path-level evidence) but uses a 70 / 10 / 20 split and validation-based baseline selection, so its numbers
are new results, not a re-run of the draft.
