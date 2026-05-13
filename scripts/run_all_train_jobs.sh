#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
CONTINUE_ON_ERROR=0
SKIP_EXISTING=1
INCLUDE_MODELS=""

usage() {
  cat <<'EOF'
Usage: scripts/run_all_train_jobs.sh [options]

Options:
  --python PATH          Python executable to use. Default: $PYTHON_BIN or python3
  --include MODELS       Comma-separated model list to run
  --no-skip-existing     Re-run stages even if outputs/.../last.ckpt already exists
  --continue-on-error    Keep running the remaining jobs if one job fails
  --dry-run              Print commands without executing them
  -h, --help             Show this help message

Examples:
  scripts/run_all_train_jobs.sh
  scripts/run_all_train_jobs.sh --include cnn_b0ns,sed_v2s
  scripts/run_all_train_jobs.sh --no-skip-existing --continue-on-error
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --include)
      INCLUDE_MODELS="$2"
      shift 2
      ;;
    --no-skip-existing)
      SKIP_EXISTING=0
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

mapfile -t JOBS < <(
  "$PYTHON_BIN" - <<'PY'
import importlib
import os

preferred_models = [
    "sed_v2s",
    "sed_b3ns",
    "sed_seresnext26t",
    "cnn_v2s",
    "cnn_resnet34d",
    "cnn_b3ns",
    "cnn_b0ns",
]
preferred_stages = [
    "pretrain_ce",
    "pretrain_bce",
    "train_ce",
    "train_bce",
    "finetune",
]

for model_name in preferred_models:
    cfg = importlib.import_module(f"configs.{model_name}").basic_cfg
    available_stages = (
        set(cfg.seed.keys())
        & set(cfg.epochs.keys())
        & set(cfg.lr.keys())
        & set(cfg.model_ckpt.keys())
        & set(cfg.output_path.keys())
        & set(cfg.loss.keys())
    )
    for stage in preferred_stages:
        if stage in available_stages:
            output_dir = cfg.output_path[stage]
            print(f"{model_name}\t{stage}\t{output_dir}")
PY
)

if [[ -n "$INCLUDE_MODELS" ]]; then
  IFS=',' read -r -a INCLUDED <<< "$INCLUDE_MODELS"
else
  INCLUDED=()
fi

contains_model() {
  local candidate="$1"
  if [[ ${#INCLUDED[@]} -eq 0 ]]; then
    return 0
  fi
  local item
  for item in "${INCLUDED[@]}"; do
    if [[ "$item" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

total_jobs=0
run_jobs=0

for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r model_name stage output_dir <<< "$job"
  if ! contains_model "$model_name"; then
    continue
  fi
  total_jobs=$((total_jobs + 1))
done

if [[ "$total_jobs" -eq 0 ]]; then
  echo "No matching jobs found."
  exit 0
fi

echo "Discovered $total_jobs training jobs."

for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r model_name stage output_dir <<< "$job"
  if ! contains_model "$model_name"; then
    continue
  fi

  checkpoint_path="$output_dir/last.ckpt"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$checkpoint_path" ]]; then
    echo "[skip] $model_name $stage ($checkpoint_path exists)"
    continue
  fi

  run_jobs=$((run_jobs + 1))
  cmd=( "$PYTHON_BIN" train.py --stage "$stage" --model_name "$model_name" )

  echo "[run $run_jobs/$total_jobs] ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
    if ! "${cmd[@]}"; then
      echo "[fail] $model_name $stage" >&2
    fi
  else
    "${cmd[@]}"
  fi
done

echo "Done."
