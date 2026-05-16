#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

/home/nasneo/miniforge3/envs/BirdSound/bin/python \
  scripts/package_kaggle_openvino_2026.py \
  --clean \
  --overwrite-export \
  "$@"
