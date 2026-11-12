#!/usr/bin/env bash
# Full study: download, preprocess, train, all experiments, report. Several CPU-hours; resumable.
# EPIC_WORKERS sets the number of parallel worker processes (default: value in each config).
set -euo pipefail
cd "$(dirname "$0")/.."
for d in movielens-1m lastfm; do epic-rec download "$d"; done
epic-rec prepare --config configs/movielens.yaml
epic-rec prepare --config configs/lastfm.yaml
epic-rec train --config configs/movielens.yaml
epic-rec train --config configs/lastfm.yaml
epic-rec pipeline --config configs/experiments/pipeline_full.yaml
echo "report: results/report"
