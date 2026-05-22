#!/usr/bin/env bash
# reproduce Camelyon16 CSV experiment.
# 5-fold CV on train split, evaluation on held-out test, repeated 5 runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  Running: Camelyon16"
echo "=========================================="
python scripts/run_camelyon16_csv.py --config configs/camelyon16_csv.yaml

echo ""
echo "Camelyon16 experiment complete."
