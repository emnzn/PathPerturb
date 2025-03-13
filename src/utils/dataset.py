import os
from typing import (
    Dict, 
    Tuple, 
    Literal, 
    Callable, 
    Optional,
    Any
)

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

def get_split_dir(
    name: Literal["pcam", "gleason-grading"], 
    split: Literal["train", "val", "test"], 
    data_dir: str, 
    encoder: Literal["uni", "gigapath", "virchow"] = None, 
    embedding_mode: bool = False
    ) -> str:
    
    if embedding_mode:
        split_dir = os.path.join(data_dir, name, encoder, split)

    else:
        split_dir = os.path.join(data_dir, name, split)

    return split_dir

def get_transform_fn(
    name: str,
    embedding_mode: bool,
    custom_transform_fn: Optional[Callable] = None
    ) -> Callable:
    
    if custom_transform_fn is not None:
        return custom_transform_fn

    if embedding_mode:
        return embedding_transform_fn
    

    if name == "pcam":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    if name == "gleason-grading":
        return img_transform_fn


def get_dataset(
    name: Literal["pcam", "gleason-grading"],
    split: Literal["train", "val", "test"],
    data_dir: str,
    embedding_mode: bool = False,
    encoder: Literal["uni", "gigapath", "virchow"] = None,
    custom_transform_fn: Optional[Callable] = None
    ) -> Dataset:

    """
    Gets returns the dataset class for the selected dataset and split.

    Parameters
    ----------
    name: Literal["pcam", "gleason-grading"]
    """

    valid_datasets = ["pcam", "gleason-grading"]
    if name not in valid_datasets: raise ValueError(f"name must be one of {valid_datasets}")

    split_dir = get_split_dir(
        data_dir=data_dir,
        name=name,
        split=split,
        encoder=encoder,
        embedding_mode=embedding_mode
    )
    
    if not os.path.isdir(split_dir): 
        raise InvalidSplitError(f"{split} set cannot be found in path {os.path.dirname(split_dir)}.")
    
    transform_fn = get_transform_fn(name, embedding_mode, custom_transform_fn)

    if name == "pcam":
        if embedding_mode:
            dataset = deeplake.open_read_only(data_dir).pytorch(transform=transform_fn)

        else:
            dataset = PCAM(data_dir, split=split, transform=transform_fn)

    if name == "gleason-grading":
        dataset = deeplake.open_read_only(split_dir).pytorch(transform=transform_fn)
    
    return dataset