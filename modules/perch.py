import csv
import hashlib
from pathlib import Path

import numpy as np


def feature_file_stem(filename):
    normalized = str(filename).replace("\\", "/")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    stem = Path(normalized).stem
    return f"{stem}-{digest}"


class PerchFeatureStore:
    def __init__(self, root, embedding_dim=1280, required=False):
        self.root = Path(root)
        self.embedding_dim = int(embedding_dim)
        self.required = bool(required)
        self.by_window = {}
        self.by_file = {}
        self.available = self.root.exists()

        if not self.available:
            if self.required:
                raise FileNotFoundError(f"Perch feature directory does not exist: {self.root}")
            print(f"warning: Perch feature directory does not exist, distill loss will be skipped: {self.root}")
            return

        index_path = self.root / "index.csv"
        if index_path.exists():
            self._load_index(index_path)

    def _load_index(self, index_path):
        with index_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename")
                feature_path = row.get("path") or row.get("embedding_path")
                if not filename or not feature_path:
                    continue
                path = Path(feature_path)
                if not path.is_absolute():
                    path = self.root / path
                if not path.exists():
                    continue

                end_sec = row.get("end_sec")
                if end_sec not in (None, ""):
                    self.by_window[(filename, int(round(float(end_sec))))] = path
                else:
                    self.by_file[filename] = path

    def _candidate_paths(self, filename, end_sec=None):
        stem = feature_file_stem(filename)
        features_dir = self.root / "features"
        if end_sec is not None:
            end_sec = int(round(float(end_sec)))
            yield features_dir / f"{stem}__{end_sec}.npy"
            yield self.root / f"{stem}__{end_sec}.npy"
        yield features_dir / f"{stem}.npy"
        yield self.root / f"{stem}.npy"

    def _load_path(self, path):
        arr = np.load(path).astype(np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
        return arr

    def load(self, filename, end_sec=None):
        path = None
        if end_sec is not None:
            path = self.by_window.get((filename, int(round(float(end_sec)))))
        if path is None:
            path = self.by_file.get(filename)
        if path is not None:
            return self._load_path(path)

        for candidate in self._candidate_paths(filename, end_sec=end_sec):
            if candidate.exists():
                return self._load_path(candidate)
        return None

    def load_many(self, filename, end_secs):
        features = []
        for end_sec in end_secs:
            feature = self.load(filename, end_sec=end_sec)
            if feature is not None:
                features.append(feature)
        if not features:
            feature = self.load(filename, end_sec=None)
            if feature is not None:
                features.append(feature)

        if not features:
            return np.zeros(self.embedding_dim, dtype=np.float32), False

        feature = np.stack(features).mean(axis=0).astype(np.float32)
        if feature.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Perch feature dim mismatch for {filename}: "
                f"expected {self.embedding_dim}, got {feature.shape[0]}"
            )
        return feature, True
