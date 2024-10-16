from pathlib import Path
from torch.utils.data import Dataset
from typing import List
import torch
import json


def load_dataset(data_dir: Path, files: List[Path]):
    data = []
    for file in files:
        with open(data_dir / file, "r") as f:
            data.extend(f.readlines())
    data = [json.loads(d) for d in data]
    return data


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
