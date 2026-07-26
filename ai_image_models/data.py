from pathlib import Path

import kagglehub
import numpy as np
import torch
from torch.utils.data import Dataset


class PixelArtDataset(Dataset):
    def __init__(self):
        data_dir = Path(kagglehub.dataset_download("ebrahimelgazar/pixel-art"))
        images = np.load(data_dir / "sprites.npy", allow_pickle=False)
        labels = np.load(data_dir / "sprites_labels.npy", allow_pickle=False)
        self.x = torch.from_numpy(images).float() / 255.0
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        images = np.load(data_dir / "sprites.npy", allow_pickle=False)
        labels = np.load(data_dir / "sprites_labels.npy", allow_pickle=False)
        self.x = torch.from_numpy(images).float() / 255.0
        self.y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]