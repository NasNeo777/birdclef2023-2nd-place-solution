from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import openvino as ov
if not hasattr(ov, "Core"):
    import openvino.runtime as ov
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm


DATA_DIR = Path(os.environ.get("BIRDCLEF_DATA_DIR", "/kaggle/input/competitions/birdclef-2026"))
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "/kaggle/input/birdclef2026-2nd-place-openvino"))
OUT_PATH = Path(os.environ.get("OUT_PATH", "/kaggle/working/submission.csv"))
OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "CPU")
DRY_RUN_FILES = int(os.environ.get("DRY_RUN_FILES", "2"))
DRY_RUN_AUDIO_DIR = os.environ.get("DRY_RUN_AUDIO_DIR", "").strip()

SAMPLE_RATE = 32000
SCORE_WINDOW_SECONDS = 5.0
CONTEXT_WINDOW_SECONDS = 10.0
MODEL_BATCH_FALLBACK = 120


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    xml_dir: str
    source_key: str
    use_delta: bool
    crop: tuple[int, int] | None
    output_type: str
    ensemble_weight: float
    tta_delta: int | None = None


MODEL_SPECS = [
    ModelSpec(
        name="sed_v2s",
        kind="sed",
        xml_dir="outputs/sed_v2s/openvino",
        source_key="spec384",
        use_delta=True,
        crop=None,
        output_type="prob",
        ensemble_weight=0.25,
        tta_delta=3,
    ),
    ModelSpec(
        name="sed_seresnext26t",
        kind="sed",
        xml_dir="outputs/sed_seresnext26t/openvino",
        source_key="spec256",
        use_delta=False,
        crop=None,
        output_type="prob",
        ensemble_weight=0.10,
        tta_delta=2,
    ),
    ModelSpec(
        name="sed_b3ns",
        kind="sed",
        xml_dir="outputs/sed_b3ns/openvino",
        source_key="spec300",
        use_delta=True,
        crop=None,
        output_type="prob",
        ensemble_weight=0.21,
        tta_delta=3,
    ),
    ModelSpec(
        name="cnn_resnet34d",
        kind="cnn",
        xml_dir="outputs/cnn_resnet34d/openvino",
        source_key="spec256",
        use_delta=True,
        crop=(128, 384),
        output_type="logit",
        ensemble_weight=0.10,
    ),
    ModelSpec(
        name="cnn_b3ns",
        kind="cnn",
        xml_dir="outputs/cnn_b3ns/openvino",
        source_key="spec300_80",
        use_delta=True,
        crop=(150, 450),
        output_type="logit",
        ensemble_weight=0.15,
    ),
    ModelSpec(
        name="cnn_v2s",
        kind="cnn",
        xml_dir="outputs/cnn_v2s/openvino",
        source_key="spec_rev2s",
        use_delta=True,
        crop=(250, 750),
        output_type="logit",
        ensemble_weight=0.15,
    ),
    ModelSpec(
        name="cnn_b0ns",
        kind="cnn",
        xml_dir="outputs/cnn_b0ns/openvino",
        source_key="spec256",
        use_delta=False,
        crop=(128, 384),
        output_type="logit",
        ensemble_weight=0.04,
    ),
]

SPEC_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}


def selected_model_specs() -> list[ModelSpec]:
    raw = os.environ.get("MODEL_NAMES", "").strip()
    if not raw:
        chosen = MODEL_SPECS
    else:
        chosen = [SPEC_BY_NAME[name.strip()] for name in raw.split(",") if name.strip()]
    raw_weights = os.environ.get("MODEL_WEIGHTS", "").strip()
    if raw_weights:
        weights = [float(x.strip()) for x in raw_weights.split(",") if x.strip()]
        if len(weights) != len(chosen):
            raise ValueError("MODEL_WEIGHTS must match MODEL_NAMES in length")
        chosen = [
            ModelSpec(
                name=spec.name,
                kind=spec.kind,
                xml_dir=spec.xml_dir,
                source_key=spec.source_key,
                use_delta=spec.use_delta,
                crop=spec.crop,
                output_type=spec.output_type,
                ensemble_weight=weight,
                tta_delta=spec.tta_delta,
            )
            for spec, weight in zip(chosen, weights)
        ]
    return chosen


def list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    return sorted(path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in exts)


def parse_end_second(row_id: str, stem: str) -> int | None:
    prefix = f"{stem}_"
    if not row_id.startswith(prefix):
        return None
    try:
        return int(row_id[len(prefix) :])
    except ValueError:
        return None


def end_seconds_for_file(sample: pd.DataFrame, stem: str) -> list[int]:
    values = [parse_end_second(str(row_id), stem) for row_id in sample["row_id"].astype(str)]
    parsed = sorted(value for value in values if value is not None)
    return parsed if parsed else list(range(5, 65, 5))


def find_target_audio_dir() -> tuple[Path, list[Path], bool]:
    test_dir = DATA_DIR / "test_soundscapes"
    test_files = list_audio_files(test_dir)
    if test_files:
        return test_dir, test_files, False

    candidate_dirs: list[Path] = []
    if DRY_RUN_AUDIO_DIR:
        candidate_dirs.append(Path(DRY_RUN_AUDIO_DIR))
    candidate_dirs.extend([DATA_DIR / "train_soundscapes", DATA_DIR / "unlabeled_soundscapes"])
    for candidate in candidate_dirs:
        files = list_audio_files(candidate)
        if files:
            return candidate, files[: max(1, DRY_RUN_FILES)], True
    raise RuntimeError(
        "No hidden test_soundscapes audio files were available. "
        "Set DRY_RUN_AUDIO_DIR or run this notebook in Kaggle submission mode."
    )


def read_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def extract_context_clip(audio: np.ndarray, end_sec: int) -> np.ndarray:
    scored_start = float(end_sec) - SCORE_WINDOW_SECONDS
    context_pad = 0.5 * (CONTEXT_WINDOW_SECONDS - SCORE_WINDOW_SECONDS)
    start_sec = scored_start - context_pad
    end_sec_context = float(end_sec) + context_pad
    start = int(round(start_sec * SAMPLE_RATE))
    end = int(round(end_sec_context * SAMPLE_RATE))
    clip_length = int(round(CONTEXT_WINDOW_SECONDS * SAMPLE_RATE))
    clip = np.zeros(clip_length, dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(end, len(audio))
    if src_end > src_start:
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        clip[dst_start:dst_end] = audio[src_start:src_end]
    return clip


def compute_deltas(specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate") -> torch.Tensor:
    device = specgram.device
    dtype = specgram.dtype
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])
    n = (win_length - 1) // 2
    denom = n * (n + 1) * (2 * n + 1) / 3
    specgram = F.pad(specgram, (n, n), mode=mode)
    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)
    output = F.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    return output.reshape(shape)


def make_delta(x: torch.Tensor) -> torch.Tensor:
    x = x.transpose(3, 2)
    x = compute_deltas(x)
    x = x.transpose(3, 2)
    return x


def image_delta(x: torch.Tensor) -> torch.Tensor:
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    return torch.cat([x, delta_1, delta_2], dim=1)


@lru_cache(maxsize=None)
def mel_ops(hop_length: int, n_mels: int, f_min: float, f_max: float, n_fft: int):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        n_fft=n_fft,
        center=True,
        pad_mode="constant",
        norm="slaney",
        onesided=True,
        mel_scale="slaney",
    )
    db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return mel, db


def build_multiscale_specs(audio_clips: np.ndarray) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(audio_clips).float().unsqueeze(1)

    hop_384 = int(SCORE_WINDOW_SECONDS * SAMPLE_RATE) // (384 - 1)
    hop_256 = int(SCORE_WINDOW_SECONDS * SAMPLE_RATE) // (256 - 1)
    hop_300 = int(SCORE_WINDOW_SECONDS * SAMPLE_RATE) // (300 - 1)

    mel_384, db_384 = mel_ops(hop_384, 128, 0, SAMPLE_RATE // 2, 2048)
    mel_256, db_256 = mel_ops(hop_256, 128, 0, SAMPLE_RATE // 2, 2048)
    mel_300, db_300 = mel_ops(hop_300, 128, 50, 14000, 1024)
    mel_rev2s, db_rev2s = mel_ops(320, 64, 50, 14000, 1024)

    spec_384 = (db_384(mel_384(x)) + 80.0) / 80.0
    spec_256 = db_256(mel_256(x)) / 255.0
    spec_300 = db_300(mel_300(x)) / 255.0
    spec_300_80 = (spec_300 * 255.0 + 80.0) / 80.0
    spec_rev2s = (db_rev2s(mel_rev2s(x)) + 80.0) / 80.0

    return {
        "spec384": spec_384,
        "spec256": spec_256,
        "spec300": spec_300,
        "spec300_80": spec_300_80,
        "spec_rev2s": spec_rev2s,
    }


def apply_model_route(specs: dict[str, torch.Tensor], model_spec: ModelSpec) -> np.ndarray:
    x = specs[model_spec.source_key]
    if model_spec.use_delta:
        x = image_delta(x)
    if model_spec.crop is not None:
        start, end = model_spec.crop
        x = x[:, :, :, start:end]
    return x.cpu().numpy().astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_np(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def resolve_xml_path(model_spec: ModelSpec) -> Path:
    model_dir = WEIGHTS_ROOT / model_spec.xml_dir
    expected = model_dir / f"{model_spec.name}.xml"
    if expected.exists():
        return expected
    candidates = sorted(model_dir.glob("*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No OpenVINO XML found for {model_spec.name} under {model_dir}")
    return candidates[0]


def load_optional_class_names(default_target_cols: list[str]) -> list[str]:
    explicit = os.environ.get("CLASS_NAMES_PATH", "").strip()
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(
        [
            WEIGHTS_ROOT / "class_names.txt",
            WEIGHTS_ROOT / "bird_cols.txt",
            WEIGHTS_ROOT / "labels.json",
        ]
    )
    for path in candidates:
        if not path or not path.exists():
            continue
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                for key in ("class_names", "labels", "bird_cols"):
                    value = data.get(key)
                    if isinstance(value, list) and value:
                        return [str(x) for x in value]
            if isinstance(data, list) and data:
                return [str(x) for x in data]
        else:
            rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            if rows:
                return rows
    return list(default_target_cols)


def align_to_submission(pred: np.ndarray, classes: list[str], target_cols: list[str]) -> np.ndarray:
    if classes == target_cols:
        return pred.astype(np.float32, copy=False)
    out = np.full((pred.shape[0], len(target_cols)), 1e-6, dtype=np.float32)
    lookup = {label: idx for idx, label in enumerate(classes)}
    for j, label in enumerate(target_cols):
        if label in lookup:
            out[:, j] = pred[:, lookup[label]]
    return out


@dataclass
class Runner:
    spec: ModelSpec
    infer_request: Any
    input_names: list[str]
    max_batch: int
    class_names: list[str]

    def predict(self, batch: np.ndarray) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(batch), self.max_batch):
            chunk = batch[start : start + self.max_batch].astype(np.float32, copy=False)
            chunk_len = len(chunk)
            if chunk_len < self.max_batch:
                pad_shape = (self.max_batch - chunk_len, *chunk.shape[1:])
                chunk = np.concatenate([chunk, np.zeros(pad_shape, dtype=np.float32)], axis=0)
            if self.spec.kind == "sed":
                if len(self.input_names) >= 2:
                    result = self.infer_request.infer(
                        {
                            self.input_names[0]: chunk,
                            self.input_names[1]: np.array(self.spec.tta_delta, dtype=np.int64),
                        }
                    )
                else:
                    result = self.infer_request.infer({self.input_names[0]: chunk})
            else:
                result = self.infer_request.infer({self.input_names[0]: chunk})
            output = next(iter(result.values()))
            outputs.append(np.asarray(output, dtype=np.float32)[:chunk_len])
        pred = np.concatenate(outputs, axis=0)
        if self.spec.output_type == "prob":
            pred = logit_np(pred)
        return pred


def build_runner(model_spec: ModelSpec, class_names: list[str], core: ov.Core) -> Runner:
    xml_path = resolve_xml_path(model_spec)
    compiled = core.compile_model(str(xml_path), OPENVINO_DEVICE)
    infer_request = compiled.create_infer_request()
    input_names = [port.get_any_name() for port in compiled.inputs]
    try:
        max_batch = int(compiled.input(0).shape[0])
    except Exception:
        try:
            max_batch = int(compiled.input(0).partial_shape[0])
        except Exception:
            max_batch = MODEL_BATCH_FALLBACK
    if max_batch <= 0:
        max_batch = MODEL_BATCH_FALLBACK
    return Runner(
        spec=model_spec,
        infer_request=infer_request,
        input_names=input_names,
        max_batch=max_batch,
        class_names=class_names,
    )


def predict_file(runners: list[Runner], audio: np.ndarray, end_seconds: list[int], target_cols: list[str]) -> np.ndarray:
    clips = np.stack([extract_context_clip(audio, end_sec) for end_sec in end_seconds]).astype(np.float32, copy=False)
    specs = build_multiscale_specs(clips)

    merged: np.ndarray | None = None
    total_weight = 0.0
    for runner in runners:
        model_input = apply_model_route(specs, runner.spec)
        pred = runner.predict(model_input)
        if pred.shape[1] != len(runner.class_names):
            raise RuntimeError(
                f"{runner.spec.name} output dimension {pred.shape[1]} does not match class list length {len(runner.class_names)}"
            )
        pred = align_to_submission(pred, runner.class_names, target_cols)
        weight = runner.spec.ensemble_weight
        merged = pred * weight if merged is None else merged + pred * weight
        total_weight += weight
    if merged is None or total_weight <= 0:
        raise RuntimeError("No model predictions were produced")
    return sigmoid_np(merged / total_weight).astype(np.float32, copy=False)


def run_submission() -> None:
    sample_path = DATA_DIR / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found under {DATA_DIR}")
    sample = pd.read_csv(sample_path)
    target_cols = [col for col in sample.columns if col != "row_id"]
    class_names = load_optional_class_names(target_cols)

    target_dir, test_files, is_dry_run = find_target_audio_dir()
    if is_dry_run:
        print(f"hidden test_soundscapes not available; dry-run on {len(test_files)} files from {target_dir}", flush=True)

    selected_specs = selected_model_specs()
    print("models:", [spec.name for spec in selected_specs], flush=True)
    print("weights_root:", WEIGHTS_ROOT, flush=True)
    core = ov.Core()
    runners = [build_runner(spec, class_names, core) for spec in selected_specs]

    all_rows: list[str] = []
    all_preds: list[np.ndarray] = []
    started = time.time()
    for path in tqdm(test_files, total=len(test_files), desc="soundscapes"):
        audio = read_audio(path, SAMPLE_RATE)
        end_seconds = end_seconds_for_file(sample, path.stem)
        row_ids = [f"{path.stem}_{end_sec}" for end_sec in end_seconds]
        pred = predict_file(runners, audio, end_seconds, target_cols)
        all_rows.extend(row_ids)
        all_preds.append(pred)

    values = np.concatenate(all_preds, axis=0)
    sub = pd.DataFrame(values, columns=target_cols)
    sub.insert(0, "row_id", all_rows)

    sample_ids = sample["row_id"].astype(str).tolist()
    if len(sub) == len(sample) and set(sub["row_id"].astype(str)) == set(sample_ids):
        sub = sub.set_index("row_id").loc[sample_ids].reset_index()

    sub = sub[["row_id"] + target_cols]
    sub[target_cols] = sub[target_cols].clip(0.0, 1.0).astype(np.float32)
    assert not sub.isna().any().any()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_PATH, index=False)
    mode = "dry-run" if is_dry_run else "submission"
    print("saved", OUT_PATH, sub.shape, mode, f"{(time.time() - started) / 60:.2f} min")
    print(sub.iloc[:3, :8])


if __name__ == "__main__":
    run_submission()
