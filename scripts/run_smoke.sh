#!/usr/bin/env bash
# CPU smoke run on synthetic data: tests, demo, smoke experiment and report (a few minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
pytest -q -m "not network"
epic-rec demo --out results/demo
epic-rec pipeline --config configs/experiments/pipeline_smoke.yaml --n-boot 500
echo "report: results/synthetic/report"
