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
MODELS_CSV="${MODELS_CSV:-sed_v2s,sed_seresnext26t,sed_b3ns,cnn_resnet34d,cnn_b3ns,cnn_v2s,cnn_b0ns}"
SKIP_PERCH="${SKIP_PERCH:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
CREATE_PERCH_ENV="${CREATE_PERCH_ENV:-1}"
SKIP_PERCH_INSTALL="${SKIP_PERCH_INSTALL:-0}"
PERCH_NUM_PROCESSES="${PERCH_NUM_PROCESSES:-4}"
PERCH_CPU_MASK="${PERCH_CPU_MASK:-0-7}"
PERCH_THREADS_PER_PROC="${PERCH_THREADS_PER_PROC:-2}"
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
  --models LIST             Comma-separated models. Default: sed_v2s,sed_seresnext26t,sed_b3ns,cnn_resnet34d,cnn_b3ns,cnn_v2s,cnn_b0ns
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
  --perch-num-processes N   Split extraction into N parallel CPU processes (default: 4)
  --perch-overwrite         Regenerate existing Perch .npy files
  --no-create-perch-env     Do not create the Perch TensorFlow conda env
  --skip-perch-install      Do not install/update Perch env dependencies
  --train-extra ARGS        Extra quoted args passed to scripts/run_all_train_jobs.sh
  --package-extra ARGS      Extra quoted args passed to package_kaggle_openvino_2026.py
  -h, --help                Show help

Useful environment overrides:
  PERCH_NUM_PROCESSES=4          Split extraction across N parallel CPU processes (default: 4)
  PERCH_CPU_MASK=0-7             CPU affinity mask for Perch processes (default: 0-7, P-cores only)
  PERCH_THREADS_PER_PROC=2       Override per-process TF thread count (default: 2)
  BIRDSOUND_PYTHON=/path/to/train/python
  PERCH_KAGGLE_HANDLE=google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
  PIP_RETRIES=10
  PIP_TIMEOUT=120
  EXPORT_BATCH_SIZE=12

Example:
  # Single-process extraction (default)
  scripts/run_perch_train_export.sh --perch-batch-size 16
  # Multi-process sharding: split extraction across 4 CPU processes
  scripts/run_perch_train_export.sh --perch-num-processes 4 --perch-batch-size 16
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
    --perch-num-processes) PERCH_NUM_PROCESSES="$2"; shift 2 ;;
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
  local num_procs="${PERCH_NUM_PROCESSES:-1}"

  if [[ "$num_procs" -le 1 ]]; then
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
    return
  fi

  # ── Multi-process sharding ──────────────────────────────────────────
  local shard_dir="$PERCH_OUTPUT_DIR/.shards_$$"
  local log_dir="$PERCH_OUTPUT_DIR/.logs_$$"
  mkdir -p "$shard_dir" "$log_dir"

  # Split manifest into N shards, preserving CSV header in each
  echo "[perch-extract] splitting manifest into $num_procs shards..."
  run_perch_python - "$PERCH_MANIFEST" "$num_procs" "$shard_dir" <<'PY'
import csv, sys
from pathlib import Path
manifest = Path(sys.argv[1])
num = int(sys.argv[2])
shard_dir = Path(sys.argv[3])
with manifest.open() as f:
    rows = list(csv.DictReader(f))
    fields = list(rows[0].keys()) if rows else ["filename", "path", "end_sec"]
chunk = (len(rows) + num - 1) // num
for i in range(num):
    part = rows[i * chunk:(i + 1) * chunk]
    out = shard_dir / f"shard_{i}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(part)
    print(f"[perch-extract] shard {i}: {len(part)} jobs -> {out}", file=sys.stderr)
PY

  # Limit per-process threads to avoid CPU thrashing.
  # On hybrid architectures (P-cores + E-cores), set PERCH_THREADS_PER_PROC
  # manually because nproc counts E-cores which are slow for TF inference.
  local threads_per_proc
  if [[ -n "${PERCH_THREADS_PER_PROC:-}" ]]; then
    threads_per_proc="$PERCH_THREADS_PER_PROC"
  else
    local total_cores
    total_cores=$(nproc 2>/dev/null || echo 1)
    threads_per_proc=$(( (total_cores + num_procs - 1) / num_procs ))
  fi
  [[ "$threads_per_proc" -lt 1 ]] && threads_per_proc=1
  echo "[perch-extract] num_procs=$num_procs threads_per_proc=$threads_per_proc"

  local pids=()
  local shard_i

  # Count total jobs and pre-existing .npy files for progress tracking
  local total_jobs
  total_jobs=$(tail -n +2 "$PERCH_MANIFEST" 2>/dev/null | wc -l)
  local features_dir="$PERCH_OUTPUT_DIR/features"
  mkdir -p "$features_dir"

  for ((shard_i = 0; shard_i < num_procs; shard_i++)); do
    local shard_manifest="$shard_dir/shard_${shard_i}.csv"
    local shard_log="$log_dir/shard_${shard_i}.log"
    local index_name="index_${shard_i}.csv"

    local cmd=(scripts/extract_perch_v2_features.py
      --manifest "$shard_manifest"
      --output-dir "$PERCH_OUTPUT_DIR"
      --index-name "$index_name"
      --batch-size "$PERCH_BATCH_SIZE"
      --audio-workers "$PERCH_AUDIO_WORKERS"
      --kaggle-handle "$PERCH_KAGGLE_HANDLE")
    if [[ -n "$PERCH_MODEL_DIR" ]]; then
      cmd+=(--model-dir "$PERCH_MODEL_DIR")
    fi
    if [[ "$PERCH_OVERWRITE" == "1" ]]; then
      cmd+=(--overwrite)
    fi

    echo "[perch-extract] launching shard $shard_i: ${cmd[*]}"

    TF_NUM_INTRAOP_THREADS="$threads_per_proc" \
      TF_NUM_INTEROP_THREADS="$threads_per_proc" \
      OMP_NUM_THREADS="$threads_per_proc" \
      run_perch_python "${cmd[@]}" > "$shard_log" 2>&1 &
    local child_pid=$!
    pids+=($child_pid)

    if [[ -n "${PERCH_CPU_MASK:-}" ]]; then
      # Split the CPU mask range evenly among shards and pin via PID.
      local mask_start mask_end mask_count
      IFS=- read -r mask_start mask_end <<< "$PERCH_CPU_MASK"
      mask_start=${mask_start:-0}
      mask_end=${mask_end:-$mask_start}
      mask_count=$((mask_end - mask_start + 1))
      local per_shard=$((mask_count / num_procs))
      [[ "$per_shard" -lt 1 ]] && per_shard=1
      local shard_start=$((mask_start + shard_i * per_shard))
      local shard_end=$((shard_start + per_shard - 1))
      [[ "$shard_end" -gt "$mask_end" ]] && shard_end="$mask_end"
      taskset -cp "${shard_start}-${shard_end}" "$child_pid" 2>/dev/null || true
      echo "[perch-extract]   shard $shard_i (pid $child_pid) cpu affinity: ${shard_start}-${shard_end}"
    fi
  done

  # Background progress monitor — counts .npy files every 30s
  local start_ts=$SECONDS
  local monitor_pid
  (
    while true; do
      sleep 30
      local alive=0
      for _pid in "${pids[@]}"; do
        kill -0 "$_pid" 2>/dev/null && alive=$((alive + 1))
      done
      [[ "$alive" -eq 0 ]] && break
      local done_npy
      done_npy=$(ls -f "$features_dir" 2>/dev/null | wc -l)
      done_npy=$((done_npy - 2))  # subtract . and ..
      local pct=$(( done_npy * 100 / total_jobs ))
      local elapsed_m=$(( (SECONDS - start_ts) / 60 ))
      local rate=0
      if [[ "$elapsed_m" -gt 0 ]]; then
        rate=$(( done_npy / elapsed_m ))
      fi
      printf '[perch-extract] %d/%d (%d%%) | %dm elapsed | ~%d job/m | %d/%d alive\n' \
        "$done_npy" "$total_jobs" "$pct" "$elapsed_m" "$rate" "$alive" "$num_procs"
    done
  ) &
  monitor_pid=$!

  # Wait for all shard processes and check exit codes
  local failed=0
  for i in "${!pids[@]}"; do
    local pid="${pids[$i]}"
    if ! wait "$pid"; then
      echo "[perch-extract] shard $i (pid $pid) FAILED — see log: $log_dir/shard_${i}.log" >&2
      failed=1
    else
      echo "[perch-extract] shard $i (pid $pid) finished OK"
    fi
  done

  # Stop monitor and print final count
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  local final_npy
  final_npy=$(ls -f "$features_dir" 2>/dev/null | wc -l)
  final_npy=$((final_npy - 2))
  printf '[perch-extract] done: %d/%d .npy files in %dm\n' \
    "$final_npy" "$total_jobs" "$(( (SECONDS - start_ts) / 60 ))"

  if [[ "$failed" -eq 1 ]]; then
    echo "[perch-extract] one or more shard processes failed — aborting merge" >&2
    echo "[perch-extract] shard logs preserved at: $log_dir" >&2
    echo "[perch-extract] shard manifests preserved at: $shard_dir" >&2
    return 1
  fi

  # Merge per-shard index files into the final index.csv
  echo "[perch-extract] merging index files..."
  run_perch_python - "$PERCH_OUTPUT_DIR" "$num_procs" <<'PY'
import csv, sys
from pathlib import Path
output_dir = Path(sys.argv[1])
num = int(sys.argv[2])
all_rows = []
for i in range(num):
    p = output_dir / f"index_{i}.csv"
    if not p.exists():
        print(f"[perch-extract] WARNING: missing index file: {p}", file=sys.stderr)
        continue
    with p.open(newline="") as f:
        all_rows.extend(list(csv.DictReader(f)))
    p.unlink()
merged = output_dir / "index.csv"
with merged.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "end_sec", "path"])
    w.writeheader()
    w.writerows(all_rows)
print(f"[perch-extract] merged {len(all_rows)} rows into {merged}", file=sys.stderr)
PY

  # Cleanup shard manifests and logs
  rm -rf "$shard_dir"
  echo "[perch-extract] cleaned up shard manifests, logs kept at: $log_dir"
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
