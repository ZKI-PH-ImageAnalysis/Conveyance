
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class Camelyon16CSVDataset(Dataset):

    def __init__(
        self,
        manifest_csv: str,
        dataset_root: str,
        split: str = "all",
        patch_labels_root: Optional[str] = None,
    ) -> None:
        if split not in ("train", "test", "all"):
            raise ValueError("split must be 'train', 'test', or 'all'")

        self.dataset_root = dataset_root
        self.split = split
        self._patch_labels_root = patch_labels_root

        import csv
        entries: List[Tuple[str, int]] = []
        with open(manifest_csv, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)  # skip header
            for row in reader:
                if not row:
                    continue
                raw_path = row[0].strip()
                label = int(row[1].strip())
                entries.append((raw_path, label))

        self.bag_paths: List[str] = []
        self.labels: List[int] = []
        for raw_path, label in entries:
            is_test = Path(raw_path).name.startswith("test_")
            if split == "train" and is_test:
                continue
            if split == "test" and not is_test:
                continue
            resolved = self._resolve_path(raw_path)
            self.bag_paths.append(resolved)
            self.labels.append(label)


    def _resolve_path(self, raw_path: str) -> str:
        
        candidate = os.path.join(self.dataset_root, raw_path)
        if os.path.exists(candidate):
            return candidate


        parts = [p for p in Path(raw_path).parts if p != "Camelyon16"]
        stripped = os.path.join(self.dataset_root, *parts)
        if os.path.exists(stripped):
            return stripped

        raise FileNotFoundError("Could not resolve path")

    def __len__(self) -> int:
        return len(self.bag_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:

        feat = self._load_bag(self.bag_paths[idx])
        return torch.from_numpy(feat), self.labels[idx]

    def get_patch_labels(self, idx: int) -> Optional[np.ndarray]:

        if self._patch_labels_root is None:
            return None
        stem = Path(self.bag_paths[idx]).stem
        label_path = os.path.join(self._patch_labels_root, f"{stem}.npy")
        if not os.path.exists(label_path):
            return None
        return np.load(label_path)

    @staticmethod
    def _load_bag(csv_path: str) -> np.ndarray:
        
        import pandas as pd  
        feats = pd.read_csv(csv_path, header=0).values.astype(np.float32)
        if feats.ndim == 1:
            feats = feats[np.newaxis, :]  
        return feats

  
    @classmethod
    def from_class_manifests(
        cls,
        normal_csv: str,
        tumor_csv: str,
        dataset_root: str,
        split: str = "all",
    ) -> "Camelyon16CSVDataset":
        import tempfile, csv, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            tmp_path = tmp.name
            writer = csv.writer(tmp)
            writer.writerow(["0", "label"])
            for src in (normal_csv, tumor_csv):
                with open(src, newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader)  # skip header
                    writer.writerows(reader)

        try:
            ds = cls(tmp_path, dataset_root, split=split)
        finally:
            os.unlink(tmp_path)

        return ds

    @staticmethod
    def get_labels(manifest_csv: str, split: str = "all") -> List[int]:
        #here we do cheap label loading without loading the bags, for building stratified splits
        import csv
        labels = []
        with open(manifest_csv, newline="") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if not row:
                    continue
                raw_path = row[0].strip()
                label = int(row[1].strip())
                is_test = Path(raw_path).name.startswith("test_")
                if split == "train" and is_test:
                    continue
                if split == "test" and not is_test:
                    continue
                labels.append(label)
        return labels
