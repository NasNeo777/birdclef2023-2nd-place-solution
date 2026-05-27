#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import importlib
import math
import os
import sys
from itertools import groupby
from importlib import metadata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.perch import feature_file_stem


def crop_or_pad(y, length, is_train=True, start=None):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])
        n_repeats = length // len(y)
        epsilon = length % len(y)
        y = np.concatenate([y] * n_repeats + [y[:epsilon]])
    elif len(y) > length:
        if not is_train:
            start = start or 0
        else:
            start = start or np.random.randint(len(y) - length)
        y = y[start:start + length]
    return y

KAGGLE_PERCH_V2_CPU_HANDLE = "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu"
PERCH_SAMPLE_RATE = 32000
PERCH_WINDOW_SECONDS = 5
PERCH_NUM_SAMPLES = PERCH_SAMPLE_RATE * PERCH_WINDOW_SECONDS


def find_saved_model_dir(path: Path) -> Path | None:
    if (path / "saved_model.pb").is_file():
        return path
    if not path.is_dir():
        return None

    candidates = [child for child in path.iterdir() if child.is_dir() and (child / "saved_model.pb").is_file()]
    if not candidates:
        return None

    def version_key(candidate: Path) -> tuple[int, str]:
        return (int(candidate.name), candidate.name) if candidate.name.isdigit() else (-1, candidate.name)

    return sorted(candidates, key=version_key)[-1]


def cached_kaggle_model_dir(handle: str) -> Path | None:
    handle_parts = [part for part in handle.split("/") if part]
    cache_roots = []
    if os.environ.get("KAGGLEHUB_CACHE"):
        cache_roots.append(Path(os.environ["KAGGLEHUB_CACHE"]))
    cache_roots.append(Path.home() / ".cache" / "kagglehub")

    for root in cache_roots:
        model_root = root / "models" / Path(*handle_parts)
        saved_model_dir = find_saved_model_dir(model_root)
        if saved_model_dir is not None:
            return saved_model_dir
    return None


def download_kaggle_model(handle: str) -> Path:
    cached_dir = cached_kaggle_model_dir(handle)
    if cached_dir is not None:
        return cached_dir

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required when --model-dir/--onnx-path are not provided and the model is not in "
            "~/.cache/kagglehub. Install kagglehub, or pass --model-dir pointing to the Kaggle Perch V2 CPU "
            "SavedModel directory."
        ) from exc
    downloaded_dir = Path(kagglehub.model_download(handle))
    saved_model_dir = find_saved_model_dir(downloaded_dir)
    if saved_model_dir is None:
        raise SystemExit(f"No TensorFlow SavedModel found under downloaded Kaggle model directory: {downloaded_dir}")
    return saved_model_dir


def resolve_model_dir(model_dir: Path) -> Path:
    saved_model_dir = find_saved_model_dir(model_dir)
    if saved_model_dir is None:
        raise SystemExit(f"No TensorFlow SavedModel found under --model-dir: {model_dir}")
    return saved_model_dir


def package_version(name: str) -> tuple[int, ...] | None:
    try:
        raw = metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    parts = []
    for token in raw.replace("rc", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
    return tuple(parts)


def validate_tf_runtime(skip_check: bool = False) -> None:
    if skip_check:
        return
    tf_version = package_version("tensorflow")
    if tf_version is None:
        raise SystemExit(
            "tensorflow is required for the Kaggle TensorFlow2 Perch V2 CPU model. "
            "Use a separate feature-extraction environment with TensorFlow installed."
        )

    numpy_version = package_version("numpy")
    protobuf_version = package_version("protobuf")
    problems = []
    if tf_version >= (2, 21):
        if numpy_version is None or numpy_version < (1, 26):
            problems.append("tensorflow>=2.21 requires numpy>=1.26")
        if protobuf_version is None or protobuf_version < (6, 31, 1):
            problems.append("tensorflow>=2.21 requires protobuf>=6.31.1")
    if problems:
        raise SystemExit(
            "This Python environment has an incompatible TensorFlow runtime and may segfault on import.\n"
            + "\n".join(f"- {problem}" for problem in problems)
            + "\nRun this extractor from a separate Perch/TensorFlow environment, or pass --onnx-path. "
            "Do not upgrade numpy/protobuf inside the training environment."
        )


def load_tf_model(model_dir: Path, skip_runtime_check: bool = False):
    validate_tf_runtime(skip_runtime_check)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "tensorflow is required for the Kaggle TensorFlow2 Perch V2 CPU model. "
            "Install TensorFlow in the feature-extraction environment."
        ) from exc

    model = tf.saved_model.load(str(model_dir))
    signature = model.signatures.get("serving_default")
    return tf, model, signature


def load_onnx_session(onnx_path: Path, providers: list[str] | None):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "onnxruntime is required only when --onnx-path is used. "
            "Install it or use the default Kaggle TensorFlow2 model path."
        ) from exc

    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def resolve_backend(args):
    if args.onnx_path is not None:
        return "onnx", load_onnx_session(args.onnx_path, args.providers)

    model_dir = resolve_model_dir(args.model_dir) if args.model_dir else download_kaggle_model(args.kaggle_handle)
    return "tensorflow", load_tf_model(model_dir, args.skip_tf_runtime_check)


def row_windows(row, infer_duration: float) -> list[float]:
    clip_start = row.get("clip_start_sec", np.nan)
    clip_duration = row.get("clip_duration", np.nan)
    if not np.isnan(clip_start):
        duration = float(clip_duration) if not np.isnan(clip_duration) else float(row["duration"])
        count = max(1, int(math.ceil(duration / infer_duration)))
        return [float(clip_start) + (i + 1) * infer_duration for i in range(count)]

    duration = float(row["duration"])
    count = max(1, int(math.ceil(duration / infer_duration)))
    return [(i + 1) * infer_duration for i in range(count)]


def collect_jobs(cfg, include_valid: bool) -> list[tuple[str, str, float]]:
    from modules.preprocess import preprocess

    df_train, df_valid, *_ = preprocess(cfg)
    frames = [df_train]
    if include_valid and len(df_valid):
        frames.append(df_valid)

    jobs = {}
    for df in frames:
        for _, row in df.iterrows():
            filename = str(row["filename"])
            path = str(row["path"])
            for end_sec in row_windows(row, float(cfg.infer_duration)):
                jobs[(filename, int(round(end_sec)))] = (filename, path, float(end_sec))
    return list(jobs.values())


def write_manifest(path: Path, jobs: list[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "path", "end_sec"])
        writer.writeheader()
        for filename, audio_path, end_sec in jobs:
            writer.writerow({"filename": filename, "path": audio_path, "end_sec": end_sec})


def read_manifest(path: Path) -> list[tuple[str, str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [
            (str(row["filename"]), str(row["path"]), float(row["end_sec"]))
            for row in reader
        ]


def load_window(path: str, end_sec: float, sample_rate: int) -> np.ndarray:
    try:
        import librosa as lb
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for Perch V2 audio loading in the extraction environment. "
            "Run the pipeline without --skip-perch-install, or install librosa in the Perch env."
        ) from exc

    offset = max(0.0, float(end_sec) - PERCH_WINDOW_SECONDS)
    audio, orig_sr = lb.load(path, sr=None, mono=True, offset=offset, duration=PERCH_WINDOW_SECONDS)
    if orig_sr != sample_rate:
        audio = lb.resample(audio, orig_sr=orig_sr, target_sr=sample_rate, res_type="kaiser_fast")
    audio = crop_or_pad(audio, PERCH_NUM_SAMPLES, is_train=False)
    return audio.astype(np.float32)


def load_audio_file(path: str, sample_rate: int) -> np.ndarray:
    try:
        import librosa as lb
    except ImportError as exc:
        raise SystemExit(
            "librosa is required for Perch V2 audio loading in the extraction environment. "
            "Run the pipeline without --skip-perch-install, or install librosa in the Perch env."
        ) from exc

    audio, orig_sr = lb.load(path, sr=None, mono=True)
    if orig_sr != sample_rate:
        audio = lb.resample(audio, orig_sr=orig_sr, target_sr=sample_rate, res_type="kaiser_fast")
    return audio.astype(np.float32)


def slice_window(audio: np.ndarray, end_sec: float, sample_rate: int) -> np.ndarray:
    end_sample = int(round(float(end_sec) * sample_rate))
    start_sample = max(0, end_sample - PERCH_NUM_SAMPLES)
    window = audio[start_sample:end_sample]
    if len(window) == 0:
        window = np.zeros(PERCH_NUM_SAMPLES, dtype=np.float32)
    else:
        window = crop_or_pad(window, PERCH_NUM_SAMPLES, is_train=False)
    return window.astype(np.float32)


def output_path(output_dir: Path, filename: str, end_sec: float) -> Path:
    stem = feature_file_stem(filename)
    return output_dir / "features" / f"{stem}__{int(round(end_sec))}.npy"


def extract_onnx_embeddings(session, batch: list[np.ndarray]) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    audio = np.stack(batch).astype(np.float32)
    outputs = session.run(["embedding"], {input_name: audio})
    return outputs[0].astype(np.float32)


def extract_tf_embeddings(tf_backend, batch: list[np.ndarray]) -> np.ndarray:
    tf, model, signature = tf_backend
    audio = tf.convert_to_tensor(np.stack(batch).astype(np.float32))

    if signature is not None:
        _, kwargs = signature.structured_input_signature
        if kwargs:
            input_name = next(iter(kwargs))
            outputs = signature(**{input_name: audio})
        else:
            outputs = signature(audio)
    else:
        outputs = model(audio)

    if isinstance(outputs, dict):
        if "embedding" in outputs:
            embedding = outputs["embedding"]
        else:
            candidates = [value for value in outputs.values() if tuple(value.shape.as_list()[1:]) == (1536,)]
            if len(candidates) != 1:
                raise KeyError(f"TensorFlow model outputs do not contain a unique 1536-d embedding: {list(outputs)}")
            embedding = candidates[0]
    elif hasattr(outputs, "embeddings"):
        embedding = outputs.embeddings
    else:
        embedding = outputs[0]
    return embedding.numpy().astype(np.float32)


def extract_embeddings(backend, model, batch: list[np.ndarray]) -> np.ndarray:
    if backend == "onnx":
        return extract_onnx_embeddings(model, batch)
    return extract_tf_embeddings(model, batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Perch V2 embeddings for training distillation.")
    parser.add_argument("--config", default="sed_v2s", choices=["sed_v2s", "sed_seresnext26t", "cnn_resnet34d"])
    parser.add_argument("--stage", default="train_ce", choices=["pretrain_ce", "pretrain_bce", "train_ce", "train_bce", "finetune"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="Read extraction jobs from a CSV manifest instead of importing project configs.")
    parser.add_argument("--write-manifest", type=Path, default=None, help="Write a CSV manifest and exit before loading TensorFlow/ONNX.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Local Kaggle TensorFlow2 perch_v2_cpu model directory.")
    parser.add_argument("--kaggle-handle", default=KAGGLE_PERCH_V2_CPU_HANDLE)
    parser.add_argument("--onnx-path", type=Path, default=None, help="Optional local ONNX backbone path. Overrides the Kaggle TensorFlow2 model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audio-workers", type=int, default=4, help="Parallel workers for per-window audio loading/resampling when --window-loading-mode window is used. Use 0 to disable.")
    parser.add_argument("--window-loading-mode", choices=["file", "window"], default="file", help="Load each contiguous audio file once and slice windows, or load each window separately.")
    parser.add_argument("--include-valid", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--provider", action="append", dest="providers", help="ONNX Runtime provider. Can be passed more than once.")
    parser.add_argument("--skip-tf-runtime-check", action="store_true", help="Skip TensorFlow dependency compatibility checks before import.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest is not None:
        jobs = read_manifest(args.manifest)
        output_dir = args.output_dir or Path("outputs/perch_features")
    else:
        from modules.preprocess import prepare_cfg

        cfg = importlib.import_module(f"configs.{args.config}").basic_cfg
        cfg = prepare_cfg(cfg, args.stage)
        output_dir = args.output_dir or Path(getattr(cfg, "perch_feature_dir", "outputs/perch_features"))
        jobs = collect_jobs(cfg, include_valid=args.include_valid)

    if args.write_manifest is not None:
        write_manifest(args.write_manifest, jobs)
        print(f"wrote {len(jobs)} Perch V2 extraction jobs to {args.write_manifest}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "features").mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    print(
        f"[perch-extract] jobs={total} output_dir={output_dir} "
        f"batch_size={args.batch_size} audio_workers={args.audio_workers} "
        f"window_loading_mode={args.window_loading_mode} overwrite={args.overwrite}",
        file=sys.stderr,
        flush=True,
    )
    print("[perch-extract] loading model/backend...", file=sys.stderr, flush=True)
    backend, model = resolve_backend(args)
    print(f"[perch-extract] backend ready: {backend}", file=sys.stderr, flush=True)

    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise SystemExit("tqdm is required for Perch extraction progress. Install tqdm in the Perch env.") from exc

    index_path = output_dir / "index.csv"
    rows = []
    pending_audio = []
    pending_meta = []
    pending_loads = set()
    extracted = 0
    skipped = 0
    audio_workers = max(0, int(args.audio_workers))
    max_pending_loads = max(args.batch_size, audio_workers * 4)

    progress = tqdm(total=total, desc="perch-extract", unit="job", dynamic_ncols=True)

    def update_progress() -> None:
        progress.set_postfix(
            extracted=extracted,
            skipped=skipped,
            pending=len(pending_audio),
            loading=len(pending_loads),
            refresh=False,
        )
        progress.update(1)

    def flush():
        nonlocal extracted, pending_audio, pending_meta
        if not pending_audio:
            return
        embeddings = extract_embeddings(backend, model, pending_audio)
        for embedding, (filename, end_sec, path) in zip(embeddings, pending_meta):
            rel_path = path.relative_to(output_dir)
            np.save(path, embedding)
            rows.append({"filename": filename, "end_sec": int(round(end_sec)), "path": str(rel_path)})
            extracted += 1
        pending_audio = []
        pending_meta = []

    def add_loaded(filename: str, end_sec: float, path: Path, audio: np.ndarray) -> None:
        pending_audio.append(audio)
        pending_meta.append((filename, end_sec, path))
        if len(pending_audio) >= args.batch_size:
            flush()

    def load_job(filename: str, audio_path: str, end_sec: float, path: Path):
        return filename, end_sec, path, load_window(audio_path, end_sec, PERCH_SAMPLE_RATE)

    def drain_completed_loads(block: bool = False) -> None:
        if not pending_loads:
            return
        if block:
            done, _ = wait(pending_loads, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending_loads if future.done()}
        for future in done:
            pending_loads.remove(future)
            add_loaded(*future.result())
            update_progress()

    executor = ThreadPoolExecutor(max_workers=audio_workers) if audio_workers > 0 and args.window_loading_mode == "window" else None
    try:
        if args.window_loading_mode == "file":
            for audio_path, group_iter in groupby(jobs, key=lambda item: item[1]):
                group = list(group_iter)
                to_extract = []
                for filename, _, end_sec in group:
                    path = output_path(output_dir, filename, end_sec)
                    if path.exists() and not args.overwrite:
                        rows.append({"filename": filename, "end_sec": int(round(end_sec)), "path": str(path.relative_to(output_dir))})
                        skipped += 1
                        update_progress()
                    else:
                        to_extract.append((filename, end_sec, path))

                if not to_extract:
                    continue

                progress.set_postfix(
                    extracted=extracted,
                    skipped=skipped,
                    pending=len(pending_audio),
                    loading=Path(audio_path).name[:24],
                    refresh=False,
                )
                audio = load_audio_file(audio_path, PERCH_SAMPLE_RATE)
                for filename, end_sec, path in to_extract:
                    add_loaded(filename, end_sec, path, slice_window(audio, end_sec, PERCH_SAMPLE_RATE))
                    update_progress()
        else:
            for filename, audio_path, end_sec in jobs:
                path = output_path(output_dir, filename, end_sec)
                if path.exists() and not args.overwrite:
                    rows.append({"filename": filename, "end_sec": int(round(end_sec)), "path": str(path.relative_to(output_dir))})
                    skipped += 1
                    update_progress()
                    continue

                if executor is None:
                    add_loaded(filename, end_sec, path, load_window(audio_path, end_sec, PERCH_SAMPLE_RATE))
                    update_progress()
                    continue

                pending_loads.add(executor.submit(load_job, filename, audio_path, end_sec, path))
                if len(pending_loads) >= max_pending_loads:
                    drain_completed_loads(block=True)
                else:
                    drain_completed_loads(block=False)

            while pending_loads:
                drain_completed_loads(block=True)

        flush()
        progress.set_postfix(extracted=extracted, skipped=skipped, pending=0, loading=0, refresh=True)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
        progress.close()

    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "end_sec", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"[perch-extract] wrote {len(rows)} Perch V2 embeddings to {output_dir} "
        f"(extracted={extracted}, skipped={skipped})",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
