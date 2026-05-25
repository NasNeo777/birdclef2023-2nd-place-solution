#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# cnn_v2s uses 500px EfficientNetV2-S inputs; batch 64 OOMs on 16 GB GPUs.
# cnn_b3ns (EfficientNet-B3, 300px) and cnn_b0ns (EfficientNet-B0, 256px) also OOM at default batch sizes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BIRDCLEF_BATCH_SIZE_CNN_V2S="${BIRDCLEF_BATCH_SIZE_CNN_V2S:-32}"
export BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_V2S="${BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_V2S:-2}"
export BIRDCLEF_BATCH_SIZE_CNN_B3NS="${BIRDCLEF_BATCH_SIZE_CNN_B3NS:-32}"
export BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_B3NS="${BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_B3NS:-2}"
export BIRDCLEF_BATCH_SIZE_CNN_B0NS="${BIRDCLEF_BATCH_SIZE_CNN_B0NS:-64}"
export BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_B0NS="${BIRDCLEF_ACCUMULATE_GRAD_BATCHES_CNN_B0NS:-2}"

echo "=== Step 1: Pre-download timm weights ==="
python3 scripts/download_timm_weights.py

echo ""
echo "=== Step 2: Resume training ==="
bash scripts/run_all_train_jobs.sh --python /home/nasneo/miniforge3/envs/BirdSound/bin/python --no-skip-existing --continue-on-error

echo ""
echo "=== Step 3: Package Kaggle OpenVINO folder ==="
/home/nasneo/miniforge3/envs/BirdSound/bin/python scripts/package_kaggle_openvino_2026.py --clean --overwrite-export
