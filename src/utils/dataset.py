import os
from typing import Dict, Tuple, Literal, Any

import torch
import deeplake
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import PCAM

class InvalidSplitError(ValueError):
    "Raised when the selected split is invalid."
    pass

def img_transform_fn(row: Dict[str, Any]) -> Tuple[torch.Tensor]:
    """
    Performs image-level preprocessing for encoding.
    """
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = img_transform(row["image"])
    label = torch.tensor(row["label"], dtype=torch.long)
    file_key = torch.tensor(row["file_key"], dtype=torch.long)

    return img, label, file_key

def embedding_transform_fn(row: Dict[str, Any]) -> Tuple[torch.Tensor]:
    """
    Performs embedding-level preprocessing.
    """
    embedding = torch.tensor(row["embedding"])
    label = torch.tensor(row["label"], dtype=torch.long)
    file_key = torch.tensor(row["file_key"], dtype=torch.long)

    return embedding, label, file_key

def get_dataset(
    name: Literal["pcam", "gleason-grading"],
    split: Literal["train", "val", "test"],
    data_dir: str,
    embedding_mode: bool,
    encoder: Literal["uni", "gigapath", "virchow"] = None
    ) -> Dataset:

    """
    Gets returns the dataset class for the selected dataset and split.

    Parameters
    ----------
    name: Literal["pcam", "gleason-grading"]
    """

    valid_datasets = ["pcam", "gleason-grading"]
    if name not in valid_datasets: raise ValueError(f"name must be one of {valid_datasets}")

    match name:
        case "pcam":
            if embedding_mode:
                dataset = deeplake.open_read_only(data_dir).pytorch(transform=embedding_transform_fn)

            else:
                base_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                ])
                dataset = PCAM(data_dir, split=split, transform=base_transform)

        case "gleason-grading":
            transform_fn = embedding_transform_fn if embedding_mode else img_transform_fn
            
            if embedding_mode:
                split_dir = os.path.join(data_dir, name, encoder, split)

            else:
                split_dir = os.path.join(data_dir, name, split)

            if not os.path.isdir(split_dir): 
                raise InvalidSplitError(f"{split} set cannot be found in path {os.path.dirname(split_dir)}.")

            dataset = deeplake.open_read_only(split_dir).pytorch(transform=transform_fn)
    
    return dataset