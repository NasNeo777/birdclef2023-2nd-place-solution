#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Step 1: Pre-download timm weights ==="
python3 scripts/download_timm_weights.py

echo ""
echo "=== Step 2: Resume training ==="
bash scripts/run_all_train_jobs.sh --continue-on-error

echo ""
echo "=== Step 3: Package Kaggle OpenVINO folder ==="
/home/nasneo/miniforge3/envs/BirdSound/bin/python scripts/package_kaggle_openvino_2026.py --clean --overwrite-export
