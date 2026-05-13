# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Train a single model stage
python3 train.py --stage <stage> --model_name <model_name>
# stages: pretrain_ce, pretrain_bce, train_ce, train_bce, finetune
# models: sed_v2s, sed_b3ns, sed_seresnext26t, cnn_v2s, cnn_resnet34d, cnn_b3ns, cnn_b0ns

# Run all training jobs (7 models x 5 stages), skipping existing checkpoints
bash scripts/run_all_train_jobs.sh

# Run specific models only, re-run all, dry-run first
bash scripts/run_all_train_jobs.sh --include cnn_b0ns,sed_v2s --no-skip-existing --continue-on-error
bash scripts/run_all_train_jobs.sh --dry-run

# Convert trained model to ONNX + OpenVINO
python3 convert.py --model_name <model_name>
```

## Architecture

### Config inheritance chain

`configs/common.py` defines `common_cfg` (SimpleNamespace) with global defaults. Each model config (e.g., `configs/sed_v2s.py`) does `copy.deepcopy(common_cfg)`, then overrides model-specific settings and exports `basic_cfg`. At runtime, `train.py` calls `prepare_cfg(cfg, stage)` which mutates `cfg.bird_cols`, `cfg.DURATION`, batch sizes, and part counts based on the current stage.

**Auto-detection of 2023 vs 2026**: `common.py` checks for `sample_submission.csv` at a hardcoded path. If found, it switches `dataset_version` to "2026", changes label columns, enables `fixed_clip_mode`, and sets `valid_split = 0.1`. This determines which preprocessing path runs.

### Model dispatch

`modules/model.py` has a factory `load_model(cfg, stage, train=True)` that uses `cfg.model_type` to select the class:
- `"cnn"` → `BirdClefTrainModelCNN` / `BirdClefInferModelCNN`
- `"sed"` → `BirdClefTrainModelSED` / `BirdClefInferModelSED`

All models inherit from `BirdClefModelBase(pl.LightningModule)`, which handles mel spectrogram conversion, audio augmentations (pitch shift, time shift, time/freq masking), optimizer setup (Adam + CosineAnnealingWarmRestarts), and loss computation.

### CNN vs SED architectures

- **CNN**: `timm backbone → GeM pooling → Multi-sample Dropout (5x, p=0.5) → Linear head`. During training, runs 5 forward passes with different dropout masks and averages; during validation, single pass. For long audio, the input is split into parts processed independently then recombined.
- **SED**: `BatchNorm → timm backbone (as encoder) → channel smoothing → FC → AttBlockV2`. The `AttBlockV2` computes class-wise attention over time, producing frame-level, segment-level (max-pooled), and clip-level (attention-weighted) outputs. TTA shifts the attention window by ±delta frames at inference.

### Multi-stage training pipeline

Each stage loads a checkpoint from the previous stage (`cfg.model_ckpt[stage]`). When class counts differ between stages (e.g., pretrain ~1148 species → train 206 species), `load_model()` drops incompatible weights (head, attention layers) via `strict=False` filtering. The finetune stage freezes the backbone and only trains the head.

### Audio augmentation split

Audio augmentations run in two layers:
1. **`torch-audiomentations`** (GPU, via `cfg.wave_transforms`): pitch shift, time shift, colored noise — applied inside the LightningModule on GPU tensors
2. **`audiomentations`** (CPU, via `cfg.am_audio_transforms` + `cfg.np_audio_transforms`): background noise injection, gain, noise — applied in the Dataset `__getitem__` on numpy arrays

### torch-audiomentations compat patch

`modules/model.py` monkey-patches `torchaudio.set_audio_backend` and `torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE` before importing `torch_audiomentations`, because newer torchaudio removed these globals that `torch_audiomentations` 0.11 expects.

### Data preprocessing

`modules/preprocess.py` has two paths:
- **`preprocess_2023()`**: loads `train.csv` with primary/secondary label columns, filters by `bird_cols`, splits by filename
- **`preprocess_2026()`**: loads soundscape labels + train_audio metadata, builds per-second label frames, caches audio durations, combines sources with configurable validation split

### Weighted sampling

`get_train_dataloader()` uses `WeightedRandomSampler` where sample weight = `1 / sqrt(class_count)` plus a foreground multiplier (10x for samples containing any bird species). This handles extreme class imbalance in bird audio data.

### Inference ensemble

The Kaggle inference notebooks load all 7 OpenVINO models and average predictions with fixed weights: SED_v2s=0.25, SED_seresnext=0.10, SED_b3ns=0.21, CNN_resnet34d=0.10, CNN_b3ns=0.15, CNN_v2s=0.15, CNN_b0ns=0.04.

## Key files

| File | Purpose |
|------|---------|
| `train.py` | Training entry point — parses args, loads config, preprocesses data, runs PL Trainer |
| `convert.py` | Exports PyTorch→ONNX→OpenVINO (always uses `train_ce` stage to build the model) |
| `configs/common.py` | Global config, dataset version auto-detection, species lists |
| `configs/<model>.py` | Per-model hyperparams, augmentation chains, checkpoint paths |
| `modules/model.py` | All model classes, mixup, attention block, metrics |
| `modules/dataset.py` | Dataset with pseudo-label logic, weighted sampling |
| `modules/preprocess.py` | Data loading for both 2023 and 2026 formats |
| `modules/augmentations.py` | Custom CPU-side audio augmentation classes |
| `scripts/run_all_train_jobs.sh` | Orchestrator for all training jobs with skip/continue logic |

## Important paths

- Checkpoints: `outputs/<model_name>/pytorch/<stage>/last.ckpt`
- ONNX: `outputs/<model_name>/onnx/<model_name>.onnx`
- OpenVINO: `outputs/<model_name>/openvino/`
- Training data: `inputs/train_audios/`, `inputs/train.csv`
- Background noise: `inputs/background_noise/<7 categories>/`
- Test audio: `inputs/test_audios/`
- Audio duration cache: `outputs/cache/train_audio_durations_2026.csv`
- W&B API key: set in `configs/common.py` (currently hardcoded; replace with your own)
