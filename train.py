import argparse
import importlib
from modules.preprocess import preprocess,prepare_cfg
from modules.dataset import get_train_dataloader
from modules.model import load_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning, EarlyStopping
import torch
import os
import gc
import json
from datetime import datetime
from modules.pseudo import load_pseudo_labels

def make_parser():
    parser = argparse.ArgumentParser(description='parser')
    parser.add_argument('--stage', choices=["pretrain_ce","pretrain_bce","train_ce","train_bce","finetune"])
    parser.add_argument('--model_name', choices=["sed_v2s",'sed_b3ns','sed_seresnext26t','cnn_v2s','cnn_resnet34d','cnn_b3ns','cnn_b0ns'])
    parser.add_argument('--use_pseudo', action='store_true')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    stage = args.stage
    model_name = args.model_name
    use_pseudo = args.use_pseudo
    cfg = importlib.import_module(f'configs.{model_name}').basic_cfg
    cfg = prepare_cfg(cfg,stage)
    os.environ['WANDB_API_KEY'] = cfg.WANDB_API_KEY
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        getattr(cfg, "cuda_alloc_conf", "expandable_segments:True"),
    )
    torch.set_float32_matmul_precision(getattr(cfg, "val_matmul_precision", "high"))

    pl.seed_everything(cfg.seed[stage], workers=True)

    df_train, df_valid, df_label_train, df_label_valid, sample_weight, transforms = preprocess(cfg)

    pseudo = None

    if use_pseudo:
        # =========================================================
        pseudo_file = os.path.join(cfg.pseudo_label_path, 'pseudo.json')
        hand_label_file = os.path.join(cfg.hand_label_path, 'hand_label.json')

        if not os.path.exists(pseudo_file):
            print(f"Warning: Pseudo label files not found at {cfg.pseudo_label_path}")
            use_pseudo = False
            pseudo = None
        else:
            pseudo = load_pseudo_labels(
                pseudo_file,
                hand_label_file if os.path.exists(hand_label_file) else None,
            )
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

    logger = WandbLogger(project='BirdClef-2023', name=f'{model_name}_{stage}')
    os.makedirs(cfg.output_path[stage], exist_ok=True)
    done_path = os.path.join(cfg.output_path[stage], "done.json")
    if os.path.exists(done_path):
        os.remove(done_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="validation C-MAP score pad 5",
        dirpath= cfg.output_path[stage],
        filename="best",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last= True,
        save_weights_only=True,
        verbose= True,
        every_n_epochs=1,
        mode='max'
    )
    callbacks_to_use = [checkpoint_callback]
    model = load_model(cfg,stage)
    trainer = pl.Trainer(
        devices=1,
        val_check_interval=1.0,
        deterministic=None,
        max_epochs=cfg.epochs[stage],
        num_sanity_val_steps=getattr(cfg, "num_sanity_val_steps", 0),
        accumulate_grad_batches=getattr(cfg, "accumulate_grad_batches", 1),
        gradient_clip_val=getattr(cfg, "gradient_clip_val", 0.0),
        gradient_clip_algorithm=getattr(cfg, "gradient_clip_algorithm", "norm"),
        logger=logger,
        callbacks=callbacks_to_use,
        precision=cfg.PRECISION, accelerator="auto",
    )

    print("Running trainer.fit")
    trainer.fit(model, train_dataloaders = dl_train, val_dataloaders = dl_val)

    done_payload = {
        "model_name": model_name,
        "stage": stage,
        "completed_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "epochs": cfg.epochs[stage],
        "best_model_path": checkpoint_callback.best_model_path or "",
        "best_model_score": (
            float(checkpoint_callback.best_model_score.item())
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "last_model_path": checkpoint_callback.last_model_path or "",
    }
    with open(done_path, "w") as f:
        json.dump(done_payload, f, indent=2)

    gc.collect()
    torch.cuda.empty_cache()
    return

if __name__=='__main__':
    main()
