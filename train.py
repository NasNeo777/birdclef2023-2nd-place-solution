import argparse
import importlib
from pathlib import Path
import warnings
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_SILENT", "true")

from modules.preprocess import preprocess, prepare_cfg
from modules.dataset import get_train_dataloader
from modules.model import load_model
from modules.config_names import MODEL_NAMES
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
import torch
import wandb
import gc
import json

warnings.filterwarnings(
    "ignore",
    message=r".*had to be resampled from .* hz to .* hz\. This hurt execution time\.",
    module=r"audiomentations\.core\.audio_loading_utils",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Transforms now expect an `output_type` argument.*",
    module=r"torch_audiomentations\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Argument 'onesided' has been deprecated.*",
    module=r"torchaudio\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\) is deprecated.*",
    module=r"pytorch_lightning\.utilities\._pytree",
)

torch.set_float32_matmul_precision("medium")


def env_override(prefix, model_name, stage):
    model_name = model_name.upper()
    stage = stage.upper()
    for key in (
            f"{prefix}_{model_name}_{stage}",
            f"{prefix}_{model_name}",
            prefix,
    ):
        if key in os.environ:
            return key, os.environ[key]
    return None, None


def int_env_override(prefix, model_name, stage, default):
    key, value = env_override(prefix, model_name, stage)
    if value is None:
        return default
    value = int(value)
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    print(f"{key} override: {default} -> {value}")
    return value


def precision_env_override(prefix, model_name, stage, default):
    key, value = env_override(prefix, model_name, stage)
    if value is None:
        return default
    precision = int(value) if value.isdigit() else value
    print(f"{key} override: {default} -> {precision}")
    return precision


def bool_env_override(prefix, model_name, stage, default):
    key, value = env_override(prefix, model_name, stage)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        result = True
    elif value in {"0", "false", "no", "off"}:
        result = False
    else:
        raise ValueError(f"{key} must be a boolean-like value, got {value!r}")
    print(f"{key} override: {default} -> {result}")
    return result


def float_env_override(prefix, model_name, stage, default):
    key, value = env_override(prefix, model_name, stage)
    if value is None:
        return default
    result = float(value)
    print(f"{key} override: {default} -> {result}")
    return result


def resolve_repo_path(repo_root, path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def make_parser():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--stage', required=True,
                        choices=["pretrain_ce", "pretrain_bce", "train_ce", "train_bce", "finetune"])
    parser.add_argument('--model_name', required=True,
                        choices=list(MODEL_NAMES))
    parser.add_argument('--use_pseudo', action='store_true')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    stage = args.stage
    model_name = args.model_name
    use_pseudo = args.use_pseudo
    cfg = importlib.import_module(f'configs.{model_name}').basic_cfg
    cfg.batch_size = int_env_override("BIRDCLEF_BATCH_SIZE", model_name, stage, int(cfg.batch_size))
    cfg.PRECISION = precision_env_override("BIRDCLEF_PRECISION", model_name, stage, cfg.PRECISION)
    cfg.use_llrd = False
    cfg = prepare_cfg(cfg, stage)
    accumulate_grad_batches = int_env_override(
        "BIRDCLEF_ACCUMULATE_GRAD_BATCHES",
        model_name,
        stage,
        1,
    )
    os.environ['WANDB_API_KEY'] = cfg.WANDB_API_KEY

    if use_pseudo and not getattr(cfg, "allow_pseudo", True):
        raise ValueError("Pseudo-label training is not implemented for the 2026 soundscape dataset")

    seed = int(cfg.seed[stage]) % (2 ** 32)
    pl.seed_everything(seed, workers=True)

    df_train, df_valid, df_label_train, df_label_valid, sample_weight, transforms = preprocess(cfg)

    pseudo = None

    if use_pseudo:
        # =========================================================
        pseudo_path = resolve_repo_path(repo_root, cfg.pseudo_label_path) / "pseudo.json"
        hand_label_path = resolve_repo_path(repo_root, cfg.hand_label_path) / "hand_label.json"

        with pseudo_path.open() as f:
            pseudo = json.loads(f.read())

        with hand_label_path.open() as f:
            hand_label = json.loads(f.read())

        if "subset1" not in pseudo:
            raise ValueError(f"{pseudo_path} does not contain subset1 pseudo labels")
        if "pred" not in hand_label:
            raise ValueError(f"{hand_label_path} does not contain pred labels")
        if "2023" not in hand_label["pred"]:
            raise ValueError(f"{hand_label_path} does not contain 2023 labels")

        for version in hand_label['pred'].keys():
            for filename in hand_label['pred'][version].keys():
                for label in hand_label['pred'][version][filename].keys():
                    for second in hand_label['pred'][version][filename][label].keys():
                        for i in range(len(pseudo['subset1']['pseudo'])):
                            if second in pseudo['subset1']['pseudo'][i]['pred'][version][filename][label].keys():
                                pseudo['subset1']['pseudo'][i]['pred'][version][filename][label][second] = \
                                hand_label['pred'][version][filename][label][second]
        # =========================================================

    dl_train, dl_val, ds_train, ds_val = get_train_dataloader(
        df_train,
        df_valid,
        df_label_train,
        df_label_valid,
        sample_weight,
        cfg,
        pseudo,
        transforms
    )

    logger = WandbLogger(
        project=f'BirdClef-SoftLoss-perch_7_model-{cfg.dataset_version}',
        name=f'{model_name}_{stage}',
        settings=wandb.Settings(quiet=True, console="off"),
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_roc_auc',
        dirpath=cfg.output_path[stage],
        filename='best',
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        verbose=True,
        every_n_epochs=1,
        mode='max',
    )
    callbacks_to_use = [checkpoint_callback]
    model = load_model(cfg, stage)
    trainer = pl.Trainer(
        devices=1,
        val_check_interval=1.0,
        deterministic=None,
        max_epochs=cfg.epochs[stage],
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks_to_use,
        precision=cfg.PRECISION,
        accelerator="auto",
        accumulate_grad_batches=accumulate_grad_batches,
    )

    print("Running trainer.fit")
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_val)

    del dl_train, dl_val, ds_train, ds_val, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    main()
