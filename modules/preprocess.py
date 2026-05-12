import pandas as pd
import numpy as np
from ast import literal_eval
import torch
import os

def _resolve_stage_value(value, stage, name):
    if isinstance(value, dict):
        if stage not in value:
            raise KeyError(f"{name} is missing a value for stage '{stage}'")
        value = value[stage]
    return value

def prepare_cfg(cfg,stage):
    if stage in ["pretrain_ce","pretrain_bce"]:
        cfg.bird_cols = cfg.bird_cols_pretrain
    elif stage in ["train_ce","train_bce","finetune"]:
        cfg.bird_cols = cfg.bird_cols_train
    else:
        raise NotImplementedError

    if stage == 'finetune':
        cfg.DURATION = cfg.DURATION_FINETUNE
        cfg.freeze = True
    elif stage in ["pretrain_ce","pretrain_bce","train_ce","train_bce"]:
        cfg.DURATION = cfg.DURATION_TRAIN
    else:
        raise NotImplementedError

    cfg.batch_size = int(_resolve_stage_value(cfg.batch_size, stage, "batch_size"))
    cfg.accumulate_grad_batches = int(
        _resolve_stage_value(
            getattr(cfg, "accumulate_grad_batches", 1),
            stage,
            "accumulate_grad_batches",
        )
    )

    cfg.test_batch_size = int(
        np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
    )
    cfg.test_batch_size = min(
        cfg.test_batch_size, getattr(cfg, "max_valid_batch_size", cfg.test_batch_size)
    )
    cfg.train_part = int(cfg.DURATION / cfg.infer_duration)
    cfg.active_seed = cfg.seed[stage]
    return cfg

def train_test_split(df, df_labels, cfg):
    valid_ratio = float(getattr(cfg, "valid_ratio", 0.0))
    if valid_ratio <= 0.0 or len(df) < 2:
        df_valid = pd.DataFrame(columns=df.columns)
        df_labels_valid = pd.DataFrame(columns=df_labels.columns)
        return (
            df.reset_index(drop=True),
            df_valid,
            df_labels.reset_index(drop=True),
            df_labels_valid,
        )

    rng = np.random.default_rng(getattr(cfg, "active_seed", 0))
    group_keys = df[cfg.primary_label_col].fillna("unknown").astype(str).values
    grouped_indices = {}
    for idx, group_key in enumerate(group_keys):
        grouped_indices.setdefault(group_key, []).append(idx)

    valid_indices = []
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            continue
        n_valid = int(round(len(indices) * valid_ratio))
        n_valid = max(1, n_valid)
        n_valid = min(len(indices) - 1, n_valid)
        shuffled = rng.permutation(indices)
        valid_indices.extend(shuffled[:n_valid].tolist())

    if not valid_indices:
        valid_indices = [int(rng.integers(0, len(df)))]

    valid_mask = np.zeros(len(df), dtype=bool)
    valid_mask[valid_indices] = True
    train_mask = ~valid_mask

    if not train_mask.any():
        valid_mask[valid_indices[-1]] = False
        train_mask = ~valid_mask

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    df_labels_train = df_labels.loc[train_mask].reset_index(drop=True)
    df_labels_valid = df_labels.loc[valid_mask].reset_index(drop=True)
    return df_train, df_valid, df_labels_train, df_labels_valid

def _infer_bird_cols_from_df(df, cfg):
    bird_cols = []
    seen = set()

    def add_label(label):
        if not isinstance(label, str):
            return
        label = label.strip()
        if not label or label == "soundscape" or label in seen:
            return
        seen.add(label)
        bird_cols.append(label)

    for primary_label in df[cfg.primary_label_col].values:
        add_label(primary_label)

    for secondary_labels in df[cfg.secondary_labels_col].values:
        if not isinstance(secondary_labels, (list, tuple, set)):
            continue
        for secondary_label in secondary_labels:
            add_label(secondary_label)

    return bird_cols

def preprocess(cfg):
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)
        audio = cfg.am_audio_transforms(audio,sample_rate=cfg.SR)
        return audio

    # primary_label_2023: used in birdclef2023, containing wrong label
    # primary_label_very_strict: original label from xeno-canto
    # primary_label_strict: fuse the ebird_code with same name but different number. ex: categr1	to categr
    # primary_label: fuse the same species with different ebird code: ['grbcam1',  'blkkit3',  'whcshr1', 'barowl8','barowl7','egwtea1','foxsp1','euhgul1']
    df = pd.read_csv(cfg.train_data)
    for col in ['secondary_labels', 'secondary_labels_2023', 'secondary_labels_strict', 'secondary_labels_very_strict']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    if getattr(cfg, "dynamic_bird_cols", False):
        dynamic_bird_cols = _infer_bird_cols_from_df(df, cfg)
        if dynamic_bird_cols:
            cfg.bird_cols_train = dynamic_bird_cols
            cfg.bird_cols_pretrain = dynamic_bird_cols
            cfg.bird_cols = dynamic_bird_cols
    
    df['version'] = df['collection'].astype(str) if 'collection' in df.columns else '2023'
    df['rating'] = df['rating'].mask(np.isnan(df['rating'].values),df.get('q', pd.Series([np.nan]*len(df))).map({'A':5,'B':4,'C':3,'D':2,'E':1,'no score':0}))
    if 'filename' not in df.columns:
        df['filename'] = df['id'].apply(lambda x: f'XC{x}.ogg')
    
    # In user dataset, filename already contains the path inside train_audio/
    df['path'] = df['filename'].apply(lambda x: os.path.join(cfg.train_dir, x))
    
    # Handle missing duration
    if 'duration' not in df.columns:
        print("warning: 'duration' column not found, setting default to 30.0s")
        df['duration'] = 30.0

    # ensure all the train data is available
    if not df['path'].apply(lambda x:os.path.exists(x)).all():
        print('===========================================================')
        print('warning: missing audio files in ./inputs/train_audios')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df = df[df['path'].apply(lambda x:os.path.exists(x))].reset_index(drop=True)

    labels = np.zeros(shape=(len(df),len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels,columns=cfg.bird_cols)
    include_in_train = []
    presence_type = []
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        include = False
        presence = 'background' if primary_label!='soundscape' else 'soundscape'
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i,primary_label] = 1
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i,secondary_label] = cfg.secondary_label
        presence_type.append(presence)
        include_in_train.append(include)

    df['presence_type'] = presence_type
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)

    df_labels[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)
    df = df[((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')].reset_index(drop=True)

    df_train,df_valid,df_labels_train,df_labels_valid = train_test_split(df,df_labels,cfg)
    print(f"train samples: {len(df_train)}, valid samples: {len(df_valid)}")

    class_sample_count = {col:0 for col in cfg.bird_cols}
    for primary_label, secondary_labels in zip(df_train[cfg.primary_label_col].values,df_train[cfg.secondary_labels_col].values):
        if primary_label in cfg.bird_cols:
            class_sample_count[primary_label] += 1
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                class_sample_count[secondary_label] += cfg.secondary_label_weight

    sample_weight = np.zeros(shape=(len(df_train,)))
    for i,(primary_label, secondary_labels) in enumerate(zip(df_train[cfg.primary_label_col].values,df_train[cfg.secondary_labels_col].values)):
        if primary_label in cfg.bird_cols:
            sample_weight[i] = 1.0/(class_sample_count[primary_label])
        else:
            secondary_labels_include = [secondary_label for secondary_label in secondary_labels if secondary_label in cfg.bird_cols]
            secondary_weights = [
                1.0 / class_sample_count[secondary_label]
                for secondary_label in secondary_labels_include
                if class_sample_count[secondary_label] > 0
            ]
            sample_weight[i] = np.mean(secondary_weights) if secondary_weights else 1.0

    return df_train, df_valid, df_labels_train, df_labels_valid, sample_weight, transforms
