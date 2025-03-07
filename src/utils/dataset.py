import os
from typing import Dict, Tuple, Any

import torch
import deeplake
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import PCAM

class InvalidSplitError(ValueError):
    "Raised when the selected split is invalid."
    pass

def transform_fn(row: Dict[str, Any]) -> Tuple[torch.Tensor]:
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = img_transform(row["image"])
    label = torch.tensor(row["label"], dtype=torch.long)
    file_key = torch.tensor(row["file_key"], dtype=torch.long)

    return img, label, file_key

def get_dataset(
    name: str,
    split: str,
    data_dir: str
    ) -> Dataset:

    valid_datasets = ["pcam", "gleason-grading"]
    assert name in valid_datasets, f"name must be one of {valid_datasets}"

    if name == "pcam":
        base_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = PCAM(data_dir, split=split, transform=base_transform)

    if name == "gleason-grading":
        split_dir = os.path.join(data_dir, name, split)
        if not os.path.isdir(split_dir): 
            raise InvalidSplitError(f"{split} set cannot be found in path {os.path.dirname(split_dir)}.")

        dataset = deeplake.open(split_dir).pytorch(transform=transform_fn)
    
    return dataset