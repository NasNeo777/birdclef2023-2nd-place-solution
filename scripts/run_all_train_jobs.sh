#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
CONTINUE_ON_ERROR=0
SKIP_EXISTING=1
INCLUDE_MODELS=""

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="$(command -v python3)"
fi

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
            print(f"{model_name}\t{stage}\t{output_dir}\t{len(cfg.bird_cols_train)}\t{cfg.model_type}")
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

checkpoint_matches_classes() {
  local checkpoint_path="$1"
  local num_classes="$2"
  local model_type="$3"

  "$PYTHON_BIN" - "$checkpoint_path" "$num_classes" "$model_type" <<'PY'
import sys
import torch

checkpoint_path, num_classes, model_type = sys.argv[1], int(sys.argv[2]), sys.argv[3]
state_dict = torch.load(checkpoint_path, map_location="cpu").get("state_dict", {})

if model_type == "sed":
    key = "att_block.cla.bias"
elif model_type == "cnn":
    key = "head.bias"
else:
    print("0")
    raise SystemExit(0)

value = state_dict.get(key)
if value is None:
    print("0")
elif int(value.shape[0]) == num_classes:
    print("1")
else:
    print("0")
PY
}

total_jobs=0
run_jobs=0

for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r model_name stage output_dir num_classes model_type <<< "$job"
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
  IFS=$'\t' read -r model_name stage output_dir num_classes model_type <<< "$job"
  if ! contains_model "$model_name"; then
    continue
  fi

  checkpoint_path="$output_dir/last.ckpt"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$checkpoint_path" ]]; then
    if [[ "$(checkpoint_matches_classes "$checkpoint_path" "$num_classes" "$model_type")" == "1" ]]; then
      echo "[skip] $model_name $stage ($checkpoint_path exists and matches ${num_classes} classes)"
      continue
    fi
    echo "[rerun] $model_name $stage ($checkpoint_path exists but is incompatible with ${num_classes} classes)"
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
