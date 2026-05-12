#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
  PYTHON_BIN="python3"
fi
SKIP_DONE="${SKIP_DONE:-1}"
RUN_CONVERT="${RUN_CONVERT:-0}"
USE_PSEUDO="${USE_PSEUDO:-0}"
STAGE_ORDER=(
  "pretrain_ce"
  "pretrain_bce"
  "train_ce"
  "train_bce"
  "finetune"
)

MODELS=(
  "sed_v2s"
  "sed_b3ns"
  "sed_seresnext26t"
  "cnn_v2s"
  "cnn_resnet34d"
  "cnn_b3ns"
  "cnn_b0ns"
)

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
fi

run_stage() {
  local model="$1"
  local stage="$2"
  local ckpt_dir="outputs/${model}/pytorch/${stage}"
  local done_file="${ckpt_dir}/done.json"

  if [[ "$SKIP_DONE" == "1" && -f "$done_file" ]]; then
    echo "[skip] ${model} ${stage} -> ${done_file}"
    return 0
  fi

  echo
  echo "== train: model=${model} stage=${stage} =="
  mkdir -p "$ckpt_dir"

  if [[ "$USE_PSEUDO" == "1" ]]; then
    "$PYTHON_BIN" train.py --stage "$stage" --model_name "$model" --use_pseudo
  else
    "$PYTHON_BIN" train.py --stage "$stage" --model_name "$model"
  fi
}

enabled_stages() {
  local model="$1"
  "$PYTHON_BIN" -c "import importlib; cfg = importlib.import_module('configs.${model}').basic_cfg; stage_order = ['pretrain_ce', 'pretrain_bce', 'train_ce', 'train_bce', 'finetune']; print(' '.join([stage for stage in stage_order if stage in cfg.seed]))"
}

for model in "${MODELS[@]}"; do
  echo
  echo "================ ${model} ================"
  mapfile -t stages < <(enabled_stages "$model" | tr ' ' '\n')

  if [[ "${#stages[@]}" -eq 0 ]]; then
    echo "[skip] ${model} has no enabled stages"
    continue
  fi

  for stage in "${stages[@]}"; do
    [[ -n "$stage" ]] || continue
    run_stage "$model" "$stage"
  done

  if [[ "$RUN_CONVERT" == "1" ]]; then
    echo
    echo "== convert: model=${model} =="
    "$PYTHON_BIN" convert.py --model_name "$model"
  fi
done

echo
echo "All requested runs finished."
