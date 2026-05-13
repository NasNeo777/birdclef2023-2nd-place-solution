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
    local skip_reason
    skip_reason="$("$PYTHON_BIN" -c "import importlib, os, sys; cfg = importlib.import_module('configs.${model}').basic_cfg; cache_path = getattr(cfg, 'duration_cache_path', None); dynamic = bool(getattr(cfg, 'dynamic_bird_cols', False)); done_path = '${done_file}'; upstream = None; model_ckpt = getattr(cfg, 'model_ckpt', {}).get('${stage}'); should_skip = True; reason = done_path; 
if dynamic and cache_path and os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(done_path):
    should_skip = False
    reason = f'duration cache newer than done file: {cache_path}'
elif model_ckpt:
    upstream = model_ckpt
    if os.path.basename(upstream) == 'last.ckpt':
        best_ckpt = os.path.join(os.path.dirname(upstream), 'best.ckpt')
        if os.path.exists(best_ckpt):
            upstream = best_ckpt
    if os.path.exists(upstream) and os.path.getmtime(upstream) > os.path.getmtime(done_path):
        should_skip = False
        reason = f'upstream checkpoint newer than done file: {upstream}'
sys.stdout.write(reason if should_skip else '')")"
    if [[ -n "$skip_reason" ]]; then
      echo "[skip] ${model} ${stage} -> ${skip_reason}"
      return 0
    fi
    echo "[rerun] ${model} ${stage} because prerequisites are newer than ${done_file}"
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
  "$PYTHON_BIN" -c "import importlib; cfg = importlib.import_module('configs.${model}').basic_cfg; default_order = ['pretrain_ce', 'pretrain_bce', 'train_ce', 'train_bce', 'finetune']; stage_order = getattr(cfg, 'enabled_stages', None) or default_order; print(' '.join([stage for stage in stage_order if stage in cfg.seed]))"
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
