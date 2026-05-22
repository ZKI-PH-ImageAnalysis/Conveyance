from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import scipy.io


class BenchmarkMILDataset(Dataset):
  
    def __init__(self, mat_path: str, indices: Optional[List[int]] = None):
        super().__init__()
        d = scipy.io.loadmat(mat_path, squeeze_me=False)
        if "data" not in d:
            raise ValueError(
                f"Expected a 'data' key in {mat_path!r}, got: {list(d.keys())}"
            )
        raw = d["data"] 

        self.bags: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.inst_labels_list: List[np.ndarray] = []

        idx_iter = indices if indices is not None else range(len(raw))
        for i in idx_iter:
            mat = np.array(raw[i, 0], dtype=np.float32) 
            # last column = instance label.
            inst_lbls = mat[:, -1].astype(np.int64)       
            feat = mat[:, :-1] 
            label_raw = int(raw[i, 1].flat[0])
            #normalise: -1 to 0, 1 to 1 
            label = 1 if label_raw > 0 else 0
            self.bags.append(torch.from_numpy(feat))
            self.labels.append(label)
            self.inst_labels_list.append(inst_lbls)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.long)

    @staticmethod
    def get_labels_from_mat(mat_path: str) -> List[int]:
        d = scipy.io.loadmat(mat_path, squeeze_me=False)