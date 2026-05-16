#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_openvino_2026 import DEFAULT_MODELS, export_model, read_class_names, write_class_names


DEFAULT_OUTPUT_DIR = REPO_ROOT / "birdclef2026-2nd-place-openvino"
DEFAULT_NOTEBOOK = REPO_ROOT / "notebooks" / "2nd-place-solution-inference-kernel-2026.ipynb"
DEFAULT_SAMPLE_SUBMISSION = Path("/media/nasneo/AI/datas/bird/sample_submission.csv")
DEFAULT_KAGGLE_WEIGHTS_ROOT = "/kaggle/input/birdclef2026-2nd-place-openvino"
DEFAULT_OPENVINO_WHEEL = (
    "/kaggle/input/datasets/rulerbug/openvino/"
    "openvino-2026.1.0-21367-cp312-cp312-manylinux_2_28_x86_64.whl"
)


def model_openvino_files(model_name: str) -> tuple[Path, Path]:
    cfg = importlib.import_module(f"configs.{model_name}").basic_cfg
    openvino_dir = REPO_ROOT / cfg.openvino_path
    xml_path = openvino_dir / f"{model_name}.xml"
    bin_path = openvino_dir / f"{model_name}.bin"
    if not xml_path.exists() or not bin_path.exists():
        raise FileNotFoundError(f"{model_name}: missing OpenVINO files under {openvino_dir}")
    return xml_path, bin_path


def copy_model_ir(model_name: str, output_dir: Path) -> None:
    xml_path, bin_path = model_openvino_files(model_name)
    dst_dir = output_dir / "outputs" / model_name / "openvino"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(xml_path, dst_dir / xml_path.name)
    shutil.copy2(bin_path, dst_dir / bin_path.name)


def patch_notebook_defaults(notebook_path: Path, kaggle_weights_root: str, openvino_wheel: str) -> None:
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook.get("cells", []):
        source = cell.get("source")
        if not isinstance(source, list):
            continue
        patched: list[str] = []
        for line in source:
            if 'os.environ.setdefault("WEIGHTS_ROOT"' in line:
                line = f'os.environ.setdefault("WEIGHTS_ROOT", "{kaggle_weights_root}")\n'
            elif 'WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT"' in line:
                line = f'WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "{kaggle_weights_root}"))\n'
            elif 'os.environ.setdefault("OPENVINO_WHEEL"' in line:
                line = f'os.environ.setdefault("OPENVINO_WHEEL", "{openvino_wheel}")\n'
            elif 'subprocess.check_call([sys.executable, "-m", "pip", "install"' in line:
                line = (
                    '    subprocess.check_call([sys.executable, "-m", "pip", "install", '
                    '"--no-index", "--no-deps", wheel])\n'
                )
            patched.append(line)
        cell["source"] = patched
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


def write_readme(output_dir: Path, kaggle_weights_root: str) -> None:
    dataset_slug = kaggle_weights_root.rstrip("/").split("/")[-1]
    readme = f"""# BirdCLEF 2026 2nd Place OpenVINO Weights

Upload this folder as a Kaggle Dataset.

Recommended dataset slug:

`{dataset_slug}`

The packaged inference notebook defaults to:

`{kaggle_weights_root}`

If Kaggle gives the dataset a different path, set `WEIGHTS_ROOT` in the first notebook cell to the actual `/kaggle/input/<slug>` path.

Required structure:

```text
outputs/
  class_names.txt
  <model_name>/openvino/<model_name>.xml
  <model_name>/openvino/<model_name>.bin
```
"""
    (output_dir / "README_UPLOAD.md").write_text(readme)


def ensure_clean_output_dir(output_dir: Path, clean: bool) -> None:
    if output_dir.exists() and clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="After training, export OpenVINO models and package a Kaggle-ready inference dataset folder."
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION)
    parser.add_argument("--source-notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--kaggle-weights-root", default=DEFAULT_KAGGLE_WEIGHTS_ROOT)
    parser.add_argument("--openvino-wheel", default=DEFAULT_OPENVINO_WHEEL)
    parser.add_argument("--export-batch-size", type=int, default=12)
    parser.add_argument("--skip-export", action="store_true", help="Only package existing outputs/*/openvino files.")
    parser.add_argument("--overwrite-export", action="store_true", help="Regenerate existing ONNX/OpenVINO files.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before packaging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    source_notebook = args.source_notebook.resolve()

    if args.export_batch_size < 1:
        raise ValueError("--export-batch-size must be positive")
    if not args.sample_submission.exists():
        raise FileNotFoundError(args.sample_submission)
    if not source_notebook.exists():
        raise FileNotFoundError(source_notebook)

    if not args.skip_export:
        class_names = read_class_names(args.sample_submission)
        write_class_names(REPO_ROOT / "outputs" / "class_names.txt", class_names)
        print(f"[labels] wrote {len(class_names)} classes to outputs/class_names.txt")
        for model_name in args.models:
            export_model(
                model_name,
                overwrite=args.overwrite_export,
                export_batch_size=args.export_batch_size,
            )

    ensure_clean_output_dir(output_dir, clean=args.clean)

    class_names = read_class_names(args.sample_submission)
    write_class_names(output_dir / "outputs" / "class_names.txt", class_names)
    print(f"[package] class_names.txt ({len(class_names)} classes)")

    for model_name in args.models:
        copy_model_ir(model_name, output_dir)
        print(f"[package] {model_name}")

    packaged_notebook = output_dir / source_notebook.name
    shutil.copy2(source_notebook, packaged_notebook)
    patch_notebook_defaults(
        packaged_notebook,
        kaggle_weights_root=args.kaggle_weights_root,
        openvino_wheel=args.openvino_wheel,
    )
    write_readme(output_dir, args.kaggle_weights_root)

    print(f"[done] Kaggle upload folder: {output_dir}")


if __name__ == "__main__":
    main()
