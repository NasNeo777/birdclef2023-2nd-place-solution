import librosa as lb
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from modules.utils import crop_or_pad
from modules.pseudo import normalize_audio_key
import multiprocessing

class BirdTrainDataset(Dataset):

    def __init__(self, df, df_labels, cfg, res_type="kaiser_fast",resample=True, train = True, pseudo=None, transforms=None):
        self.cfg =cfg
        self.df = df
        self.df_labels = df_labels
        self.sr = cfg.SR
        self.n_mels = cfg.n_mels
        self.fmin = cfg.f_min
        self.fmax = cfg.f_max

        self.train = train
        self.duration = cfg.DURATION

        self.audio_length = self.duration*self.sr

        self.res_type = res_type
        self.resample = resample

        self.df["weight"] = np.clip(self.df["rating"] / self.df["rating"].max(), 0.1, 1.0)
        self.pseudo = pseudo
        self.pseudo_version_aliases = self._build_version_aliases(
            getattr(cfg, "pseudo_version_aliases", {})
        )
        self.pseudo_entries_by_version = self._build_pseudo_index(pseudo)

        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def _build_version_aliases(self, aliases):
        version_aliases = {}
        for version, candidates in aliases.items():
            if isinstance(candidates, (list, tuple, set)):
                values = [str(candidate) for candidate in candidates]
            else:
                values = [str(candidates)]
            version_aliases[str(version)] = values
        return version_aliases

    def _iter_pseudo_groups(self, pseudo):
        if not isinstance(pseudo, dict):
            return

        if "pseudo" in pseudo and isinstance(pseudo["pseudo"], list):
            yield pseudo
            return

        for group in pseudo.values():
            if isinstance(group, dict) and "pseudo" in group and isinstance(group["pseudo"], list):
                yield group

    def _build_pseudo_index(self, pseudo):
        if pseudo is None:
            return {}

        entries_by_version = {}
        for group in self._iter_pseudo_groups(pseudo):
            weights = group.get("weight", [])
            for idx, oof in enumerate(group.get("pseudo", [])):
                weight = weights[idx] if idx < len(weights) else 1.0
                thresholds = oof.get("thre", {})
                for version, file_preds in oof.get("pred", {}).items():
                    normalized_map = {
                        normalize_audio_key(file_key): file_key for file_key in file_preds.keys()
                    }
                    entries_by_version.setdefault(str(version), []).append(
                        {
                            "pred": file_preds,
                            "file_key_map": normalized_map,
                            "thre": thresholds,
                            "weight": weight,
                        }
                    )
        return entries_by_version

    def _get_pseudo_entries(self, version):
        version = str(version)
        candidates = [version] + self.pseudo_version_aliases.get(version, [])
        entries = []
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            entries.extend(self.pseudo_entries_by_version.get(candidate, []))
        return entries

    def adjust_label(self,labels,filename,sample_ends,target,version,pseudo,pseudo_weights):
        adjust_label = {label:0 for label in labels if label in self.cfg.bird_cols}
        labels_comp = list(adjust_label.keys())
        for oof,w in zip(pseudo,pseudo_weights):
          for label in labels_comp:
            preds = [oof['pred'][version][filename][label][sample_end] for sample_end in sample_ends]
            thre = oof['thre'][label]
            adjusts = np.zeros(shape=(len(preds),))
            for i,pred in enumerate(preds):
              q3,q2,q1 = thre['q3'],thre['q2'],thre['q1']
              if pred>=q3:
                adjust = 1.0
              elif pred>=q2:
                adjust = 0.9
              elif pred>=q1:
                adjust = 0.5
              else:
                adjust = 0.2
              adjusts[i] = adjust
            adjust_label[label] += w * (1-np.prod(1-adjusts))
        for label in labels_comp:
          if adjust_label[label] <= 0.6:
            adjust_label[label] = 0.01
          elif adjust_label[label]<=0.75:
            adjust_label[label] = 0.6
          target[label] = target[label] * adjust_label[label]
        return target

    def apply_pseudo_labels(self, labels, filename, sample_ends, target, version):
        adjust_label = {label: 0 for label in labels if label in self.cfg.bird_cols}
        labels_comp = list(adjust_label.keys())
        normalized_filename = normalize_audio_key(filename)
        pseudo_entries = self._get_pseudo_entries(version)

        if not pseudo_entries or not labels_comp:
            return target

        for entry in pseudo_entries:
            raw_filename = entry["file_key_map"].get(normalized_filename)
            if raw_filename is None:
                continue

            file_preds = entry["pred"].get(raw_filename, {})
            thresholds = entry["thre"]
            weight = entry["weight"]
            for label in labels_comp:
                label_preds = file_preds.get(label)
                thre = thresholds.get(label)
                if not label_preds or thre is None:
                    continue

                preds = [
                    label_preds[sample_end]
                    for sample_end in sample_ends
                    if sample_end in label_preds
                ]
                if not preds:
                    continue

                adjusts = np.zeros(shape=(len(preds),))
                for i, pred in enumerate(preds):
                    q3, q2, q1 = thre["q3"], thre["q2"], thre["q1"]
                    if pred >= q3:
                        adjust = 1.0
                    elif pred >= q2:
                        adjust = 0.9
                    elif pred >= q1:
                        adjust = 0.5
                    else:
                        adjust = 0.2
                    adjusts[i] = adjust
                adjust_label[label] += weight * (1 - np.prod(1 - adjusts))

        for label in labels_comp:
            if adjust_label[label] <= 0.6:
                adjust_label[label] = 0.01
            elif adjust_label[label] <= 0.75:
                adjust_label[label] = 0.6
            target[label] = target[label] * adjust_label[label]
        return target

    def _read_audio(self, filepath, offset=0.0, duration=None):
        try:
            audio, orig_sr = sf.read(filepath, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
        except Exception:
            audio, orig_sr = lb.load(filepath, sr=None, mono=True)
            audio = audio.astype(np.float32, copy=False)

        start = max(0, int(round(offset * orig_sr)))
        if duration is None:
            end = len(audio)
        else:
            end = start + int(round(duration * orig_sr))
        audio = audio[start:end]

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=self.sr,
                res_type=self.res_type,
            )

        return audio.astype(np.float32, copy=False), orig_sr

    def load_data(self, filepath,target,row):
        filename = row['filename']
        labels = [bird for bird in list(set([row[self.cfg.primary_label_col]] + row[self.cfg.secondary_labels_col])) if bird in self.cfg.bird_cols]
        secondary_labels = [bird for bird in row[self.cfg.secondary_labels_col] if bird in self.cfg.bird_cols]
        duration = row['duration']
        version = row['version']
        presence = row['presence_type']

        # self mixup
        self_mixup_part = 1
        if (presence!='foreground') | (len(secondary_labels)>0):
          self_mixup_part = int(self.cfg.background_duration_thre/self.duration)
        work_duration = self.duration * self_mixup_part
        work_audio_length = work_duration*self.sr

        max_offset =np.max([0,duration-work_duration])
        parts = int(duration//self.cfg.infer_duration) if duration%self.cfg.infer_duration==0 else int(duration//self.cfg.infer_duration + 1)
        ends = [(p+1)*self.cfg.infer_duration for p in range(parts)]
        pseudo_max_end = ends[-1]

        if self.train:
            offset = torch.rand((1,)).numpy()[0] * max_offset
            audio_sample, orig_sr = self._read_audio(filepath, offset=offset, duration=work_duration)

            if len(audio_sample) < work_audio_length:
                audio_sample = crop_or_pad(audio_sample, length=work_audio_length,is_train=self.train)

            audio_sample = audio_sample.reshape((self_mixup_part,-1))
            audio_sample = np.sum(audio_sample, axis=0, dtype=np.float32)
            audio_sample = np.asarray(audio_sample, dtype=np.float32)

            if self.transforms is not None:
              audio_sample = self.transforms(audio_sample)
              audio_sample = np.asarray(audio_sample, dtype=np.float32)

            if len(audio_sample) != self.audio_length:
                audio_sample = crop_or_pad(audio_sample, length=self.audio_length,is_train=self.train)

            #pseudo is made every 5s. For example, if offset=7 then the nearest_offset=5
            nearest_offset = int(np.round(offset/self.cfg.infer_duration) * self.cfg.infer_duration)
            sample_ends = [str(nearest_offset+(i+1)*self.cfg.infer_duration) for i in range(int(work_duration/self.cfg.infer_duration)) if nearest_offset+(i+1)*self.cfg.infer_duration<=pseudo_max_end]
            # use pseudo and hand label if the total duration of the audio is larger than clip duration
            if (work_duration < duration)&(self.pseudo is not None):
              target = self.apply_pseudo_labels(labels,filename,sample_ends,target,version)

        else:
            audio, orig_sr = self._read_audio(filepath, offset=0, duration=self.cfg.valid_duration)

            audio_parts = int(np.ceil(len(audio)/self.audio_length))
            audio_sample = [audio[i*self.audio_length:(i+1)*self.audio_length] for i in range(audio_parts)]

            if len(audio_sample[-1])<self.audio_length:
              audio_sample[-1] = crop_or_pad(audio_sample[-1],length=self.audio_length,is_train=self.train)

            valid_len = int(self.cfg.valid_duration/self.duration)
            if len(audio_sample)> valid_len:
              audio_sample = audio_sample[0:valid_len]
            elif len(audio_sample)<valid_len:
              diff = valid_len-len(audio_sample)
              padding = [np.zeros(shape=(self.audio_length,), dtype=np.float32)] * diff
              audio_sample += padding

            audio_sample = np.stack(audio_sample).astype(np.float32, copy=False)

            sample_end = np.min([audio_parts * self.cfg.infer_duration, pseudo_max_end])
            sample_ends = [str(sample_end-i*self.cfg.infer_duration) for i in range(valid_len) if sample_end-i*self.cfg.infer_duration>0]

            if (work_duration < duration)&(self.pseudo is not None):
              target = self.apply_pseudo_labels(labels,filename,sample_ends,target,version)

        audio_sample = torch.tensor(audio_sample[np.newaxis]).float()

        target = target.values
        if not self.train:
          target[target>0] = 1
        return audio_sample,target

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = self.df_labels.loc[idx]

        weight = self.df.loc[idx,"weight"]
        if row['presence_type']!='foreground':
            weight = weight * 0.8
        audio, target = self.load_data(self.df.loc[idx, "path"],target,row)
        target = torch.tensor(target).float()
        return audio, target , weight

def get_train_dataloader(df_train, df_valid, df_labels_train, df_labels_valid, sample_weight,cfg,pseudo=None,transforms=None):
  train_num_workers = int(getattr(cfg, "num_workers", min(4, multiprocessing.cpu_count())))
  val_num_workers = int(getattr(cfg, "val_num_workers", train_num_workers))
  sample_weight = torch.from_numpy(sample_weight)
  sampler = WeightedRandomSampler(sample_weight.type('torch.DoubleTensor'), len(sample_weight),replacement=True)

  ds_train = BirdTrainDataset(
      df_train,
      df_labels_train,
      cfg,
      train = True,
      pseudo = pseudo,
      transforms = transforms,
  )
  ds_val = BirdTrainDataset(
      df_valid,
      df_labels_valid,
      cfg,
      train = False,
      pseudo = None,
      transforms=None,
  )
  dl_train = DataLoader(
      ds_train,
      batch_size=cfg.batch_size,
      sampler=sampler,
      num_workers=train_num_workers,
      pin_memory=True,
      persistent_workers=train_num_workers > 0,
  )
  dl_val = DataLoader(
      ds_val,
      batch_size=cfg.test_batch_size,
      num_workers=val_num_workers,
      pin_memory=True,
      persistent_workers=val_num_workers > 0,
  )
  return dl_train, dl_val, ds_train, ds_val
