from ast import literal_eval
import os

import numpy as np
import pandas as pd
import soundfile as sf


def parse_time_to_seconds(value):
    hours, minutes, seconds = str(value).split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def parse_list_labels(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return literal_eval(value)


def parse_soundscape_labels(value):
    if pd.isna(value):
        return []
    labels = [label.strip() for label in str(value).split(";") if label.strip()]
    if len(labels) == 1 and labels[0].lower() in {"nocall", "soundscape"}:
        return []
    return labels


def normalize_label(value):
    if pd.isna(value):
        return None
    return str(value).strip()


def normalize_labels(values):
    normalized = []
    for value in values:
        label = normalize_label(value)
        if label:
            normalized.append(label)
    return normalized


def load_duration_cache(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        return {}
    cache_df = pd.read_csv(cache_path)
    if not {"filename", "duration"}.issubset(cache_df.columns):
        return {}
    return dict(zip(cache_df["filename"], cache_df["duration"]))


def save_duration_cache(cache_path, duration_cache):
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    cache_df = pd.DataFrame(
        sorted(duration_cache.items()), columns=["filename", "duration"]
    )
    cache_df.to_csv(cache_path, index=False)


def get_audio_duration(file_path):
    info = sf.info(file_path)
    return float(info.frames) / float(info.samplerate)


def build_label_frame(df, cfg):
    labels = np.zeros(shape=(len(df), len(cfg.bird_cols)), dtype=np.float32)
    df_labels = pd.DataFrame(labels, columns=cfg.bird_cols)
    class_sample_count = {col: 0.0 for col in cfg.bird_cols}
    include_in_train = []
    for i, row_labels in enumerate(df["labels"].values):
        filtered_labels = [label for label in row_labels if label in cfg.bird_cols]
        include = len(filtered_labels) > 0
        include_in_train.append(include)
        for label in filtered_labels:
            df_labels.loc[i, label] = 1.0
            class_sample_count[label] += 1.0
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)
    return df, df_labels, class_sample_count


def compute_sample_weights(df, cfg, class_sample_count):
    sample_weight = np.zeros(shape=(len(df),), dtype=np.float32)
    default_weight = 1.0 / max(len(df), 1)
    for i, clip_labels in enumerate(df["labels"].values):
        clip_labels = [label for label in clip_labels if label in cfg.bird_cols]
        if clip_labels:
            sample_weight[i] = np.mean(
                [1.0 / max(class_sample_count[label], 1.0) for label in clip_labels]
            )
        else:
            sample_weight[i] = default_weight
    return sample_weight


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

    if getattr(cfg, "fixed_clip_mode", False):
        cfg.DURATION = cfg.infer_duration
        cfg.valid_duration = cfg.infer_duration
        cfg.test_batch_size = max(int(cfg.batch_size), 2)
        cfg.train_part = 1
        cfg.valid_part = 1
    else:
        cfg.test_batch_size = int(
            np.max([int(cfg.batch_size / (int(cfg.valid_duration) / cfg.DURATION)), 2])
        )
        cfg.train_part = int(cfg.DURATION / cfg.infer_duration)
        cfg.valid_part = int(cfg.valid_duration / cfg.infer_duration)
    return cfg


def train_test_split(df, df_labels, cfg):
    valid_split = getattr(cfg, "valid_split", 0.0)
    if valid_split <= 0 or df.empty or "filename" not in df:
        df_train = df.reset_index(drop=True)
        df_labels_train = df_labels.reset_index(drop=True)
        df_valid = pd.DataFrame(columns=df_train.columns)
        df_labels_valid = pd.DataFrame(columns=df_labels_train.columns)
        return df_train, df_valid, df_labels_train, df_labels_valid

    filenames = df["filename"].drop_duplicates().to_numpy()
    if len(filenames) < 2:
        df_train = df.reset_index(drop=True)
        df_labels_train = df_labels.reset_index(drop=True)
        df_valid = pd.DataFrame(columns=df_train.columns)
        df_labels_valid = pd.DataFrame(columns=df_labels_train.columns)
        return df_train, df_valid, df_labels_train, df_labels_valid

    seed_values = list(getattr(cfg, "seed", {}).values())
    seed = seed_values[0] if seed_values else 42
    rng = np.random.default_rng(seed)
    valid_count = min(len(filenames) - 1, max(1, int(round(len(filenames) * valid_split))))
    valid_files = set(rng.choice(filenames, size=valid_count, replace=False).tolist())

    valid_mask = df["filename"].isin(valid_files)
    train_mask = ~valid_mask

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_valid = df.loc[valid_mask].reset_index(drop=True)
    df_labels_train = df_labels.loc[train_mask].reset_index(drop=True)
    df_labels_valid = df_labels.loc[valid_mask].reset_index(drop=True)
    return df_train, df_valid, df_labels_train, df_labels_valid


def preprocess_2023(cfg):
    df = pd.read_csv(cfg.train_data)
    df['secondary_labels'] = df['secondary_labels'].apply(parse_list_labels)
    df['secondary_labels_2023'] = df['secondary_labels_2023'].apply(parse_list_labels)
    df['secondary_labels_strict'] = df['secondary_labels_strict'].apply(parse_list_labels)
    df['secondary_labels_very_strict'] = df['secondary_labels_very_strict'].apply(parse_list_labels)
    df['version'] = df['version'].astype(str)
    df['rating'] = df['rating'].mask(np.isnan(df['rating'].values),df['q'].map({'A':5,'B':4,'C':3,'D':2,'E':1,'no score':0}))
    df['filename'] = df['id'].apply(lambda x: f'XC{x}')
    df['path'] = df['id'].apply(lambda x: os.path.join(cfg.train_dir,f'XC{x}.ogg'))
    if not df['path'].apply(lambda x:os.path.exists(x)).all():
        print('===========================================================')
        print('warning: missing audio files in ./inputs/train_audios')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df = df[df['path'].apply(lambda x:os.path.exists(x))].reset_index(drop=True)

    labels = np.zeros(shape=(len(df),len(cfg.bird_cols)))
    df_labels = pd.DataFrame(labels,columns=cfg.bird_cols)
    class_sample_count = {col:0 for col in cfg.bird_cols}
    include_in_train = []
    presence_type = []
    for i,(primary_label, secondary_labels) in enumerate(zip(df[cfg.primary_label_col].values,df[cfg.secondary_labels_col].values)):
        include = False
        presence = 'background' if primary_label!='soundscape' else 'soundscape'
        if primary_label in cfg.bird_cols:
            include = True
            presence = 'foreground'
            df_labels.loc[i,primary_label] = 1
            class_sample_count[primary_label] += 1
        for secondary_label in secondary_labels:
            if secondary_label in cfg.bird_cols:
                include = True
                df_labels.loc[i,secondary_label] = cfg.secondary_label
                class_sample_count[secondary_label] += cfg.secondary_label_weight
        presence_type.append(presence)
        include_in_train.append(include)

    df['presence_type'] = presence_type
    df = df[include_in_train].reset_index(drop=True)
    df_labels = df_labels[include_in_train].reset_index(drop=True)

    keep_mask = ((df['duration']<=cfg.background_duration_thre)&(df['presence_type']!='foreground'))|(df['presence_type']=='foreground')
    df = df[keep_mask].reset_index(drop=True)
    df_labels = df_labels[keep_mask].reset_index(drop=True)

    df_train,df_valid,df_labels_train,df_labels_valid = train_test_split(df,df_labels,cfg)

    sample_weight = np.zeros(shape=(len(df_train,)))
    for i,(primary_label, secondary_labels) in enumerate(zip(df_train[cfg.primary_label_col].values,df_train[cfg.secondary_labels_col].values)):
        if primary_label in cfg.bird_cols:
            sample_weight[i] = 1.0/(class_sample_count[primary_label])
        else:
            secondary_labels_include = [secondary_label for secondary_label in secondary_labels if secondary_label in cfg.bird_cols]
            sample_weight[i] = np.mean([1.0/class_sample_count[secondary_label] for secondary_label in secondary_labels_include])

    return df_train, df_valid, df_labels_train, df_labels_valid, sample_weight


def preprocess_2026(cfg):
    if not cfg.train_labels:
        raise ValueError("cfg.train_labels must be set for the 2026 soundscape dataset")

    df_soundscape = pd.read_csv(cfg.train_labels)
    df_soundscape["start_sec"] = df_soundscape["start"].apply(parse_time_to_seconds)
    df_soundscape["end_sec"] = df_soundscape["end"].apply(parse_time_to_seconds)
    df_soundscape["labels"] = df_soundscape["primary_label"].apply(parse_soundscape_labels)
    df_soundscape["secondary_labels"] = [[] for _ in range(len(df_soundscape))]
    df_soundscape["path"] = df_soundscape["filename"].apply(
        lambda x: os.path.join(cfg.train_dir, x)
    )
    df_soundscape["clip_start_sec"] = df_soundscape["start_sec"].astype(float)
    df_soundscape["clip_duration"] = (
        df_soundscape["end_sec"] - df_soundscape["start_sec"]
    ).astype(float)
    df_soundscape["duration"] = df_soundscape["clip_duration"]
    df_soundscape["version"] = "2026_soundscape"
    df_soundscape["presence_type"] = df_soundscape["labels"].apply(
        lambda labels: "foreground" if labels else "background"
    )
    df_soundscape["rating"] = 1.0
    df_soundscape["source"] = "soundscape"

    if not df_soundscape["path"].apply(os.path.exists).all():
        print('===========================================================')
        print('warning: missing audio files in the 2026 soundscape directory')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df_soundscape = df_soundscape[
        df_soundscape["path"].apply(os.path.exists)
    ].reset_index(drop=True)
    df_soundscape, df_soundscape_labels, soundscape_counts = build_label_frame(
        df_soundscape, cfg
    )
    (
        df_soundscape_train,
        df_soundscape_valid,
        df_soundscape_labels_train,
        df_soundscape_labels_valid,
    ) = train_test_split(df_soundscape, df_soundscape_labels, cfg)

    df_audio = pd.read_csv(cfg.train_data)
    df_audio["primary_label"] = df_audio["primary_label"].apply(normalize_label)
    df_audio["secondary_labels"] = df_audio["secondary_labels"].apply(parse_list_labels)
    df_audio["secondary_labels"] = df_audio["secondary_labels"].apply(normalize_labels)
    df_audio["labels"] = df_audio.apply(
        lambda row: normalize_labels([row["primary_label"]] + row["secondary_labels"]),
        axis=1,
    )
    df_audio["path"] = df_audio["filename"].apply(
        lambda x: os.path.join(cfg.train_audio_dir, x)
    )
    df_audio["clip_start_sec"] = np.nan
    df_audio["clip_duration"] = float(cfg.infer_duration)
    df_audio["version"] = "2026_train_audio"
    df_audio["presence_type"] = df_audio["labels"].apply(
        lambda labels: "foreground" if labels else "background"
    )
    df_audio["rating"] = df_audio["rating"].fillna(1.0).astype(float)
    df_audio["source"] = "train_audio"

    if not df_audio["path"].apply(os.path.exists).all():
        print('===========================================================')
        print('warning: missing audio files in the 2026 train_audio directory')
        print('warning: only audios available will be used for training')
        print('===========================================================')
    df_audio = df_audio[df_audio["path"].apply(os.path.exists)].reset_index(drop=True)

    duration_cache = load_duration_cache(getattr(cfg, "train_audio_duration_cache", None))
    missing_filenames = [
        filename for filename in df_audio["filename"].tolist() if filename not in duration_cache
    ]
    if missing_filenames:
        for filename, path in zip(df_audio["filename"], df_audio["path"]):
            if filename not in duration_cache:
                duration_cache[filename] = get_audio_duration(path)
        save_duration_cache(getattr(cfg, "train_audio_duration_cache", None), duration_cache)

    df_audio["duration"] = df_audio["filename"].map(duration_cache).astype(float)
    df_audio, df_audio_labels, audio_counts = build_label_frame(df_audio, cfg)

    df_train = pd.concat([df_soundscape_train, df_audio], ignore_index=True, sort=False)
    df_label_train = pd.concat(
        [df_soundscape_labels_train, df_audio_labels], ignore_index=True
    )
    df_valid = df_soundscape_valid.reset_index(drop=True)
    df_label_valid = df_soundscape_labels_valid.reset_index(drop=True)

    class_sample_count = {
        col: soundscape_counts.get(col, 0.0) + audio_counts.get(col, 0.0)
        for col in cfg.bird_cols
    }
    sample_weight = compute_sample_weights(df_train, cfg, class_sample_count)

    return df_train, df_valid, df_label_train, df_label_valid, sample_weight


def preprocess(cfg):
    def transforms(audio):
        audio = cfg.np_audio_transforms(audio)
        audio = cfg.am_audio_transforms(audio,sample_rate=cfg.SR)
        return audio

    if getattr(cfg, "dataset_version", "2023") == "2026":
        df_train, df_valid, df_labels_train, df_labels_valid, sample_weight = preprocess_2026(cfg)
    else:
        df_train, df_valid, df_labels_train, df_labels_valid, sample_weight = preprocess_2023(cfg)

    return df_train, df_valid, df_labels_train, df_labels_valid, sample_weight, transforms
