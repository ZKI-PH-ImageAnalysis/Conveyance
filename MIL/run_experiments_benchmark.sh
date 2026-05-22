#!/usr/bin/env bash
# reproduce all classic MIL benchmark experiments.
# 10-fold stratified CV × 5 runs per dataset (50 results each).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for ds in musk1 musk2 elephant fox tiger; do
    echo "=========================================="
    echo "  Running: $ds"
    echo "=========================================="
    python scripts/run_benchmark.py --config "configs/benchmark_${ds}.yaml"
done

echo ""
echo "All benchmark experiments complete."
