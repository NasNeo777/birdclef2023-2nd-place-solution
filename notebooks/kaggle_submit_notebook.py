from __future__ import annotations

import importlib
import os
import runpy
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import openvino as ov
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm.auto import tqdm


REPO_DIR = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[1]))
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "/kaggle/input/birdclef2023-weights"))
DATA_DIR = Path(os.environ.get("BIRDCLEF_DATA_DIR", "/kaggle/input/birdclef-2023"))
OUT_PATH = Path(os.environ.get("OUT_PATH", "/kaggle/working/submission.csv"))

DEFAULT_MODELS = [
    "sed_v2s",
    "sed_b3ns",
    "sed_seresnext26t",
    "cnn_v2s",
    "cnn_resnet34d",
    "cnn_b3ns",
    "cnn_b0ns",
]

MODEL_NAMES = [
    x.strip()
    for x in os.environ.get("MODEL_NAMES", ",".join(DEFAULT_MODELS)).split(",")
    if x.strip()
]
MODEL_WEIGHTS = [
    float(x.strip())
    for x in os.environ.get("MODEL_WEIGHTS", ",".join(["1.0"] * len(MODEL_NAMES))).split(",")
    if x.strip()
]
if len(MODEL_WEIGHTS) != len(MODEL_NAMES):
    raise ValueError("MODEL_WEIGHTS must have the same length as MODEL_NAMES")

OPENVINO_DEVICE = os.environ.get("OPENVINO_DEVICE", "CPU")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
DRY_RUN_FILES = int(os.environ.get("DRY_RUN_FILES", "2"))
DRY_RUN_AUDIO_DIR = os.environ.get("DRY_RUN_AUDIO_DIR", "").strip()
ENSEMBLE_MERGE = os.environ.get("ENSEMBLE_MERGE", "mean").lower()

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from modules.preprocess import prepare_cfg  # noqa: E402


def parse_end_second(row_id: str, stem: str) -> int | None:
    prefix = f"{stem}_"
    if not row_id.startswith(prefix):
        return None
    try:
        return int(row_id[len(prefix) :])
    except ValueError:
        return None


def end_seconds_for_file(sample: pd.DataFrame, stem: str) -> list[int]:
    parsed = [parse_end_second(str(row_id), stem) for row_id in sample["row_id"].astype(str)]
    ends = sorted(e for e in parsed if e is not None)
    return ends if ends else list(range(5, 65, 5))


def list_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.exists():
        return []
    exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    return sorted(p for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def crop_or_pad_eval(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), mode="constant").astype(np.float32, copy=False)
    if len(audio) > length:
        return audio[:length].astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False)


def extract_window(audio: np.ndarray, end_sec: int, sample_rate: int, window_sec: float) -> np.ndarray:
    window = int(round(window_sec * sample_rate))
    end = int(round(end_sec * sample_rate))
    start = max(0, end - window)
    clip = audio[start:end]
    return crop_or_pad_eval(clip, window)


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


def make_delta(input_tensor: torch.Tensor) -> torch.Tensor:
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


def image_delta(x: torch.Tensor) -> torch.Tensor:
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    return torch.cat([x, delta_1, delta_2], dim=1)


@lru_cache(maxsize=None)
def mel_ops(sr: int, hop_length: int, n_mels: int, f_min: float, f_max: float, n_fft: int):
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
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
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return melspec, db_transform


def audio_to_image(audio: np.ndarray, cfg: Any) -> np.ndarray:
    audio = crop_or_pad_eval(audio, int(round(cfg.infer_duration * cfg.SR)))
    x = torch.from_numpy(audio).float().unsqueeze(0)
    melspec, db_transform = mel_ops(cfg.SR, cfg.hop_length, cfg.n_mels, cfg.f_min, cfg.f_max, cfg.n_fft)
    spec = melspec(x)
    spec = db_transform(spec)
    if cfg.normal == 80:
        spec = (spec + 80) / 80
    elif cfg.normal == 255:
        spec = spec / 255
    else:
        raise NotImplementedError(f"Unsupported cfg.normal={cfg.normal}")
    spec = spec.unsqueeze(1)
    if spec.shape[-1] != cfg.img_size:
        spec = F.interpolate(spec, size=(cfg.n_mels, cfg.img_size), mode="bilinear", align_corners=False)
    if cfg.in_chans == 3:
        spec = image_delta(spec)
    elif cfg.in_chans != 1:
        spec = spec.repeat(1, cfg.in_chans, 1, 1)
    return spec.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def align_to_submission(pred: np.ndarray, classes: list[str], target_cols: list[str]) -> np.ndarray:
    out = np.full((pred.shape[0], len(target_cols)), 1e-6, dtype=np.float32)
    lookup = {label: idx for idx, label in enumerate(classes)}
    for j, label in enumerate(target_cols):
        if label in lookup:
            out[:, j] = pred[:, lookup[label]]
    return out


def merge_predictions(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    stacked = np.stack(preds, axis=0).astype(np.float32, copy=False)
    if ENSEMBLE_MERGE == "min":
        return np.min(stacked, axis=0).astype(np.float32, copy=False)
    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr = weight_arr / max(float(weight_arr.sum()), 1e-12)
    return np.sum(stacked * weight_arr[:, None, None], axis=0).astype(np.float32, copy=False)


def detect_final_stage(final_model_path: str) -> str:
    for stage in ("finetune", "train_bce", "train_ce", "pretrain_bce", "pretrain_ce"):
        token = f"/{stage}/"
        if token in final_model_path.replace("\\", "/"):
            return stage
    return "train_ce"


def resolve_xml_path(model_name: str, cfg: Any) -> Path:
    xml_path = WEIGHTS_ROOT / cfg.openvino_path / f"{model_name}.xml"
    if xml_path.exists():
        return xml_path
    openvino_dir = WEIGHTS_ROOT / cfg.openvino_path
    candidates = sorted(openvino_dir.glob("*.xml"))
    if not candidates:
        raise FileNotFoundError(f"No OpenVINO xml found for {model_name} under {openvino_dir}")
    return candidates[0]


@dataclass
class Runner:
    model_name: str
    cfg: Any
    classes: list[str]
    weight: float
    compiled_model: Any
    output: Any

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        preds: list[np.ndarray] = []
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start : start + BATCH_SIZE].astype(np.float32, copy=False)
            logits = self.compiled_model(batch)[self.output]
            preds.append(sigmoid_np(np.asarray(logits, dtype=np.float32)))
        return np.concatenate(preds, axis=0)


def build_runner(model_name: str, model_weight: float, core: ov.Core) -> Runner:
    cfg = importlib.import_module(f"configs.{model_name}").basic_cfg
    stage = detect_final_stage(cfg.final_model_path)
    cfg = prepare_cfg(cfg, stage)
    xml_path = resolve_xml_path(model_name, cfg)
    compiled_model = core.compile_model(str(xml_path), OPENVINO_DEVICE)
    output = compiled_model.output(0)
    return Runner(
        model_name=model_name,
        cfg=cfg,
        classes=[str(x) for x in cfg.bird_cols],
        weight=model_weight,
        compiled_model=compiled_model,
        output=output,
    )


def predict_file(runner: Runner, audio: np.ndarray, end_seconds: list[int], target_cols: list[str]) -> np.ndarray:
    images = np.stack(
        [
            audio_to_image(
                extract_window(audio, end_sec, runner.cfg.SR, runner.cfg.infer_duration),
                runner.cfg,
            )
            for end_sec in end_seconds
        ]
    ).astype(np.float32, copy=False)
    pred = runner.predict(images)
    return align_to_submission(pred, runner.classes, target_cols)


def run_submission() -> None:
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    target_cols = [c for c in sample.columns if c != "row_id"]
    test_dir = DATA_DIR / "test_soundscapes"
    test_files = list_audio_files(test_dir)
    is_dry_run = False
    if not test_files:
        is_dry_run = True
        candidate_dirs: list[Path] = []
        if DRY_RUN_AUDIO_DIR:
            candidate_dirs.append(Path(DRY_RUN_AUDIO_DIR))
        candidate_dirs.extend(
            [
                DATA_DIR / "train_soundscapes",
                DATA_DIR / "unlabeled_soundscapes",
            ]
        )
        for candidate_dir in candidate_dirs:
            candidate_files = list_audio_files(candidate_dir)
            if candidate_files:
                test_dir = candidate_dir
                test_files = candidate_files[: max(1, DRY_RUN_FILES)]
                print(
                    f"hidden test_soundscapes not available; dry-run on {len(test_files)} files from {test_dir}",
                    flush=True,
                )
                break
        if not test_files:
            raise RuntimeError(
                "No hidden test_soundscapes audio files found, and no dry-run audio files were available. "
                "Set DRY_RUN_AUDIO_DIR or run this notebook in Kaggle submission mode."
            )

    core = ov.Core()
    runners = [build_runner(model_name, model_weight, core) for model_name, model_weight in zip(MODEL_NAMES, MODEL_WEIGHTS)]

    all_rows: list[str] = []
    all_preds: list[np.ndarray] = []
    started = time.time()
    for path in tqdm(test_files, total=len(test_files), desc="soundscapes"):
        audio = read_audio(path, sample_rate=runners[0].cfg.SR)
        end_seconds = end_seconds_for_file(sample, path.stem)
        row_ids = [f"{path.stem}_{end_sec}" for end_sec in end_seconds]
        model_preds = [predict_file(runner, audio, end_seconds, target_cols) for runner in runners]
        pred = merge_predictions(model_preds, [runner.weight for runner in runners])
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
