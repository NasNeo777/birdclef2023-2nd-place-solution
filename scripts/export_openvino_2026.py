#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.model import load_model
from modules.preprocess import prepare_cfg


DEFAULT_MODELS = [
    "sed_v2s",
    "sed_seresnext26t",
    "sed_b3ns",
    "cnn_resnet34d",
    "cnn_b3ns",
    "cnn_v2s",
    "cnn_b0ns",
]


def final_stage(checkpoint_path: str) -> str:
    parts = Path(checkpoint_path).parts
    for stage in ("soft_loss", "finetune", "train_bce", "train_ce", "pretrain_bce", "pretrain_ce"):
        if stage in parts:
            return stage
    raise ValueError(f"Cannot infer stage from {checkpoint_path}")


def read_class_names(sample_submission: Path) -> list[str]:
    with sample_submission.open(newline="") as f:
        return next(csv.reader(f))[1:]


def write_class_names(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names) + "\n")


def torch_export(model, args, onnx_path: Path, input_names: list[str], output_names: list[str], opset: int) -> None:
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
    )
    try:
        torch.onnx.export(model, args, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, args, str(onnx_path), **kwargs)


def convert_with_openvino(onnx_path: Path, output_dir: Path, model_name: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "openvino.tools.mo",
        "--input_model",
        str(onnx_path),
        "--output_dir",
        str(output_dir),
        "--compress_to_fp16",
        "--model_name",
        model_name,
    ]
    subprocess.run(cmd, check=True)


def export_model(model_name: str, overwrite: bool, export_batch_size: int) -> None:
    cfg = importlib.import_module(f"configs.{model_name}").basic_cfg
    stage = final_stage(cfg.final_model_path)

    # The 2026 local config uses 5-second fixed clips for training data loading.
    # OpenVINO export must keep the original model attention duration, otherwise
    # SED +/- TTA slices become empty.
    cfg.fixed_clip_mode = False
    cfg = prepare_cfg(cfg, stage)

    checkpoint = Path(cfg.final_model_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"{model_name}: missing checkpoint {checkpoint}")

    onnx_dir = Path(cfg.onnx_path)
    openvino_dir = Path(cfg.openvino_path)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    openvino_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = onnx_dir / f"{model_name}.onnx"
    xml_path = openvino_dir / f"{model_name}.xml"
    bin_path = openvino_dir / f"{model_name}.bin"
    if xml_path.exists() and bin_path.exists() and not overwrite:
        print(f"[skip] {model_name}: {xml_path} already exists")
        return

    model = load_model(cfg, stage, train=False)
    input_shape = (export_batch_size, *cfg.input_shape[1:])
    dummy = torch.randn(*input_shape)

    if cfg.model_type == "sed":
        tta_delta = int(getattr(cfg, "tta_delta", 2))
        export_args = (dummy, tta_delta)
        input_names = [cfg.input_names[0]]
    else:
        export_args = dummy
        input_names = [cfg.input_names[0]]

    print(f"[onnx] {model_name}: {checkpoint} -> {onnx_path}")
    torch_export(
        model=model,
        args=export_args,
        onnx_path=onnx_path,
        input_names=input_names,
        output_names=cfg.output_names,
        opset=int(cfg.opset_version or 10),
    )

    print(f"[openvino] {model_name}: {onnx_path} -> {openvino_dir}")
    convert_with_openvino(onnx_path, openvino_dir, model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 2026 BirdCLEF checkpoints to notebook-ready OpenVINO IR.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--sample-submission", type=Path, default=Path("/media/nasneo/AI/datas/bird/sample_submission.csv"))
    parser.add_argument("--class-names-out", type=Path, default=Path("outputs/class_names.txt"))
    parser.add_argument("--export-batch-size", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_batch_size < 1:
        raise ValueError("--export-batch-size must be positive")
    if not args.sample_submission.exists():
        raise FileNotFoundError(args.sample_submission)
    class_names = read_class_names(args.sample_submission)
    write_class_names(args.class_names_out, class_names)
    print(f"[labels] wrote {len(class_names)} classes to {args.class_names_out}")

    for model_name in args.models:
        export_model(model_name, overwrite=args.overwrite, export_batch_size=args.export_batch_size)


if __name__ == "__main__":
    main()
