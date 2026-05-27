#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_BIRDSOUND_PYTHON="/home/nasneo/miniforge3/envs/BirdSound/bin/python"
BIRDSOUND_PYTHON="${BIRDSOUND_PYTHON:-$DEFAULT_BIRDSOUND_PYTHON}"
PERCH_ENV="${PERCH_ENV:-PerchV2}"
PERCH_PYTHON="${PERCH_PYTHON:-}"
PERCH_BATCH_SIZE="${PERCH_BATCH_SIZE:-16}"
PERCH_AUDIO_WORKERS="${PERCH_AUDIO_WORKERS:-8}"
PERCH_OUTPUT_DIR="${PERCH_OUTPUT_DIR:-outputs/perch_features}"
PERCH_MANIFEST="${PERCH_MANIFEST:-$PERCH_OUTPUT_DIR/all_jobs.csv}"
PERCH_MODEL_DIR="${PERCH_MODEL_DIR:-}"
PERCH_KAGGLE_HANDLE="${PERCH_KAGGLE_HANDLE:-google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu}"
MODELS_CSV="${MODELS_CSV:-sed_v2s,sed_seresnext26t,cnn_resnet34d}"
SKIP_PERCH="${SKIP_PERCH:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
CREATE_PERCH_ENV="${CREATE_PERCH_ENV:-1}"
SKIP_PERCH_INSTALL="${SKIP_PERCH_INSTALL:-0}"
PERCH_OVERWRITE="${PERCH_OVERWRITE:-0}"
PIP_RETRIES="${PIP_RETRIES:-10}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_INDEX_URL_OPT="--index-url $PIP_INDEX_URL"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
PACKAGE_EXTRA_ARGS="${PACKAGE_EXTRA_ARGS:-}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-12}"

usage() {
  cat <<'EOF'
Usage: scripts/run_perch_train_export.sh [options]

Runs the full local pipeline:
  1. Build one merged Perch V2 extraction manifest for the selected models/stages
  2. Extract Perch V2 embeddings with an isolated TensorFlow env
  3. Train selected models
  4. Export/package OpenVINO weights for Kaggle inference

Options:
  --models LIST             Comma-separated models. Default: sed_v2s,sed_seresnext26t,cnn_resnet34d
  --perch-batch-size N      Perch extraction batch size. Default: 16
  --perch-audio-workers N   Parallel audio loading/resampling workers. Default: 8
  --perch-env NAME          Conda env for TensorFlow Perch extraction. Default: PerchV2
  --perch-python PATH       Use this Python for Perch extraction instead of conda run
  --perch-model-dir PATH    Local Kaggle perch_v2_cpu model directory
  --perch-output-dir PATH   Output directory for Perch features. Default: outputs/perch_features
  --skip-perch              Skip manifest generation and Perch extraction
  --skip-train              Skip training
  --skip-export             Skip OpenVINO export, package existing files
  --skip-package            Skip packaging entirely
  --perch-overwrite         Regenerate existing Perch .npy files
  --no-create-perch-env     Do not create the Perch TensorFlow conda env
  --skip-perch-install      Do not install/update Perch env dependencies
  --train-extra ARGS        Extra quoted args passed to scripts/run_all_train_jobs.sh
  --package-extra ARGS      Extra quoted args passed to package_kaggle_openvino_2026.py
  -h, --help                Show help

Useful environment overrides:
  BIRDSOUND_PYTHON=/path/to/train/python
  PERCH_KAGGLE_HANDLE=google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  PIP_RETRIES=10
  PIP_TIMEOUT=120
  EXPORT_BATCH_SIZE=12

Example:
  scripts/run_perch_train_export.sh --perch-batch-size 16
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS_CSV="$2"; shift 2 ;;
    --perch-batch-size) PERCH_BATCH_SIZE="$2"; shift 2 ;;
    --perch-audio-workers) PERCH_AUDIO_WORKERS="$2"; shift 2 ;;
    --perch-env) PERCH_ENV="$2"; shift 2 ;;
    --perch-python) PERCH_PYTHON="$2"; shift 2 ;;
    --perch-model-dir) PERCH_MODEL_DIR="$2"; shift 2 ;;
    --perch-output-dir) PERCH_OUTPUT_DIR="$2"; PERCH_MANIFEST="$2/all_jobs.csv"; shift 2 ;;
    --skip-perch) SKIP_PERCH=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --skip-package) SKIP_PACKAGE=1; shift ;;
    --perch-overwrite) PERCH_OVERWRITE=1; shift ;;
    --no-create-perch-env) CREATE_PERCH_ENV=0; shift ;;
    --skip-perch-install) SKIP_PERCH_INSTALL=1; shift ;;
    --train-extra) TRAIN_EXTRA_ARGS="$2"; shift 2 ;;
    --package-extra) PACKAGE_EXTRA_ARGS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -x "$BIRDSOUND_PYTHON" ]]; then
  echo "BirdSound Python not found or not executable: $BIRDSOUND_PYTHON" >&2
  exit 1
fi

IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
MODEL_ARGS=("${MODELS[@]}")

run_perch_python() {
  if [[ -n "$PERCH_PYTHON" ]]; then
    PYTHONUNBUFFERED=1 "$PERCH_PYTHON" "$@"
  else
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n "$PERCH_ENV" python "$@"
  fi
}

ensure_perch_env() {
  if [[ -n "$PERCH_PYTHON" ]]; then
    echo "[perch-env] using PERCH_PYTHON=$PERCH_PYTHON"
    return
  fi

  if conda env list | awk '{print $1}' | grep -qx "$PERCH_ENV"; then
    echo "[perch-env] found conda env: $PERCH_ENV"
  else
    if [[ "$CREATE_PERCH_ENV" != "1" ]]; then
      echo "Perch env $PERCH_ENV does not exist and --no-create-perch-env was set" >&2
      exit 1
    fi
    echo "[perch-env] creating conda env: $PERCH_ENV"
    conda create -n "$PERCH_ENV" python=3.10 -y
  fi

  if [[ "$SKIP_PERCH_INSTALL" == "1" ]]; then
    echo "[perch-env] dependency install skipped"
    return
  fi

  local dep_check
  dep_check=$(conda run -n "$PERCH_ENV" python -c '
from importlib import metadata
from packaging.version import Version

required = {
    "tensorflow": ("2.20.0", "2.21.0"),
    "numpy": ("1.26.0", "2.3.0"),
    "librosa": ("0.10.0", "0.12.0"),
    "pandas": ("2.2.0", "2.4.0"),
    "scikit-learn": ("1.5.0", "1.8.0"),
    "soundfile": ("0.0.0", None),
    "scipy": ("0.0.0", None),
    "soxr": ("0.0.0", None),
    "kagglehub": ("0.0.0", None),
}
missing = []
for pkg, (min_v, max_v) in required.items():
    try:
        version = Version(metadata.version(pkg))
    except metadata.PackageNotFoundError:
        missing.append(pkg)
        continue
    if min_v and version < Version(min_v):
        missing.append(f"{pkg}>={min_v}")
    if max_v and version >= Version(max_v):
        missing.append(f"{pkg}<{max_v}")
if missing:
    print("missing_or_incompatible=" + ",".join(missing))
else:
    print("ok")
')
  if [[ "$dep_check" == *"ok"* ]]; then
    echo "[perch-env] dependencies already satisfy requirements"
    return
  fi
  echo "[perch-env] $dep_check"

  echo "[perch-env] installing TensorFlow Perch dependencies with retries"
  conda run -n "$PERCH_ENV" python -m pip install --no-input \
    --retries "$PIP_RETRIES" \
    --timeout "$PIP_TIMEOUT" \
    --resume-retries "$PIP_RETRIES" \
    $PIP_INDEX_URL_OPT \
    kagglehub \
    'tensorflow==2.20.*' \
    'numpy>=1.26,<2.3' \
    'pandas>=2.2,<2.4' \
    'scikit-learn>=1.5,<1.8' \
    'librosa>=0.10,<0.12' \
    soundfile \
    scipy \
    soxr
}

available_jobs() {
  "$BIRDSOUND_PYTHON" - "$MODELS_CSV" <<'PY'
import importlib
import sys

models = [item for item in sys.argv[1].split(',') if item]
stages = ["pretrain_ce", "pretrain_bce", "train_ce", "train_bce", "finetune"]
for model_name in models:
    cfg = importlib.import_module(f"configs.{model_name}").basic_cfg
    available = (
        set(cfg.seed)
        & set(cfg.epochs)
        & set(cfg.lr)
        & set(cfg.model_ckpt)
        & set(cfg.output_path)
        & set(cfg.loss)
    )
    for stage in stages:
        if stage in available:
            print(f"{model_name}	{stage}")
PY
}

build_perch_manifest() {
  mkdir -p "$PERCH_OUTPUT_DIR/manifests"
  local manifest_dir="$PERCH_OUTPUT_DIR/manifests"
  local combined_tmp="$manifest_dir/all_jobs.raw.csv"
  : > "$combined_tmp"

  local first=1
  while IFS=$'\t' read -r model_name stage; do
    [[ -z "${model_name:-}" ]] && continue
    local part="$manifest_dir/${model_name}_${stage}.csv"
    echo "[perch-manifest] $model_name $stage -> $part"
    "$BIRDSOUND_PYTHON" scripts/extract_perch_v2_features.py \
      --config "$model_name" \
      --stage "$stage" \
      --write-manifest "$part"
    if [[ "$first" -eq 1 ]]; then
      cat "$part" >> "$combined_tmp"
      first=0
    else
      tail -n +2 "$part" >> "$combined_tmp"
    fi
  done < <(available_jobs)

  "$BIRDSOUND_PYTHON" - "$combined_tmp" "$PERCH_MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seen = set()
rows = []
with src.open(newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['filename'], int(round(float(row['end_sec']))))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'path', 'end_sec'])
    writer.writeheader()
    writer.writerows(rows)
print(f"[perch-manifest] wrote {len(rows)} unique jobs to {dst}")
PY
}

extract_perch_features() {
  local cmd=(scripts/extract_perch_v2_features.py
    --manifest "$PERCH_MANIFEST"
    --output-dir "$PERCH_OUTPUT_DIR"
    --batch-size "$PERCH_BATCH_SIZE"
    --audio-workers "$PERCH_AUDIO_WORKERS"
    --kaggle-handle "$PERCH_KAGGLE_HANDLE")
  if [[ -n "$PERCH_MODEL_DIR" ]]; then
    cmd+=(--model-dir "$PERCH_MODEL_DIR")
  fi
  if [[ "$PERCH_OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  echo "[perch-extract] ${cmd[*]}"
  run_perch_python "${cmd[@]}"
}

run_training() {
  echo "[train] models=$MODELS_CSV"
  # shellcheck disable=SC2206
  local extra=( $TRAIN_EXTRA_ARGS )
  PYTHON_BIN="$BIRDSOUND_PYTHON" scripts/run_all_train_jobs.sh \
    --include "$MODELS_CSV" \
    "${extra[@]}"
}

run_packaging() {
  if [[ "$SKIP_PACKAGE" == "1" ]]; then
    echo "[package] skipped"
    return
  fi

  local args=(scripts/package_kaggle_openvino_2026.py
    --models "${MODEL_ARGS[@]}"
    --export-batch-size "$EXPORT_BATCH_SIZE"
    --clean)
  if [[ "$SKIP_EXPORT" == "1" ]]; then
    args+=(--skip-export)
  else
    args+=(--overwrite-export)
  fi
  # shellcheck disable=SC2206
  local extra=( $PACKAGE_EXTRA_ARGS )
  args+=("${extra[@]}")

  echo "[package] $BIRDSOUND_PYTHON ${args[*]}"
  "$BIRDSOUND_PYTHON" "${args[@]}"
}

if [[ "$SKIP_PERCH" != "1" ]]; then
  ensure_perch_env
  build_perch_manifest
  extract_perch_features
else
  echo "[perch] skipped"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  run_training
else
  echo "[train] skipped"
fi

run_packaging

echo "[done] full pipeline completed"
