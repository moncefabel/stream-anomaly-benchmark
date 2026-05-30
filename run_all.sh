#!/bin/bash
# run_all.sh
# ----------
# Full benchmark pipeline for stream-anomaly-benchmark.
# Runs all steps in order and saves all results.
#
# Usage: bash run_all.sh

set -e  # exit on first error

echo "=================================================="
echo "  stream-anomaly-benchmark — Full Pipeline"
echo "=================================================="

cd "$(dirname "$0")/src"

echo ""
echo "[1/4] Checking dataset availability..."
python -c "from datasets import dataset_summary; dataset_summary()"

echo ""
echo "[2/4] Running main benchmark (TEASER + ECONOMY-K)..."
python -c "
import sys; sys.path.insert(0, '.')
from evaluate import benchmark
import pandas as pd
DATASETS = ['ItalyPowerDemand', 'ECG200', 'SonyAIBORobotSurface1', 'ECG5000', 'Wafer']
df = benchmark(datasets=DATASETS, seeds=[42, 0, 1])
df.to_csv('../results/benchmark_results_full.csv', index=False)
print('Saved → results/benchmark_results_full.csv')
"

echo ""
echo "[3/4] Generating figures..."
python plot_results.py

echo ""
echo "[3b/4] Running sensitivity analysis..."
python sensitivity_analysis.py

echo ""
echo "[4/4] Running orchestrator (cooperation schemes)..."
python orchestrator.py

echo ""
echo "=================================================="
echo "  Done. All results in results/"
echo "=================================================="