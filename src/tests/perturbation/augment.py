import os
import sys
from pathlib import Path
from functools import partial
from typing import Any, Dict, Tuple, Literal
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import deeplake
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.constants import (
    DATA_DIR,
    ASSET_DIR,
    CONFIG_DIR,
    EMBEDDING_DIR
)

from utils import (
    get_args,
    get_dataset,
    Network,
    NetworkHandler
)

def adjust_brightness(
    img: np.ndarray, 
    alpha: float | int
    ) -> np.ndarray:

    if alpha < -1 or alpha > 1:
        raise ValueError(f"Alpha of value {alpha} is out of range. Ensure value is between -1.0 and 1.0.")
    
    if alpha > 0:
        img = img + (abs(alpha) * (255 - img))
    
    if alpha < 0:
        img = img - (abs(alpha) * (img - 0))

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def apply_perturbation(
    img: np.ndarray, 
    perturbation_type: Literal["brightness"], 
    alpha: float | int
    ):

    perturbations = {
        "brightness": adjust_brightness
    }

    if perturbation_type in perturbations:
        return perturbations[perturbation_type](img, alpha)
    
    return img

def perturbation_fn(
    row: Dict[str, Any], 
    perturbation_type: str, 
    alpha: float | int
    ) -> Tuple[torch.Tensor]:

    """
    Performs image-level preprocessing for encoding.
    """
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = apply_perturbation(
        img=row["image"], 
        perturbation_type=perturbation_type, 
        alpha=alpha
    )
    img = img_transform(img)

    label = torch.tensor(row["label"], dtype=torch.long)
    file_key = torch.tensor(row["file_key"], dtype=torch.long)

    return img, label, file_key


def main():
    arg_path = os.path.join(CONFIG_DIR, "augment.yaml")
    args = get_args(arg_path)

    encoder_dir = os.path.join(ASSET_DIR, "model-weights", "pre-trained-weights")
    dest_dir = os.path.join(EMBEDDING_DIR, args["dataset"], args["encoder"], "perturbations", args["perturbations"]["type"])
    os.makedirs(dest_dir, exist_ok=True)

    model = Network(args["encoder"], encoder_dir)
    network_handler = NetworkHandler(model)
    embedding_dim = model.fc.head.in_features

    perturbation_type = args["perturbations"]["type"]
    perturbation_start = args["perturbations"]["range"]["min"]
    perturbation_end = args["perturbations"]["range"]["max"]
    perturbation_interval = args["perturbations"]["range"]["interval"]

    for strength in np.arange(perturbation_start, perturbation_end, perturbation_interval):
        custom_transform_fn = partial(perturbation_fn, perturbation_type=perturbation_type, alpha=strength)

        dataset = get_dataset(
            name=args["dataset"], 
            split="val" if args["dataset"] != "gleason-grading" else "test", 
            data_dir=DATA_DIR,
            custom_transform_fn=custom_transform_fn
        )

        data_loader = DataLoader(
            dataset,
            batch_size=args["batch_size"],
            shuffle=False
        )

        perturbation_ds = deeplake.create(
            os.path.join(f"file://{dest_dir}", f"{perturbation_type}_{strength:.4f}")
        )

        perturbation_ds.add_column("embedding", dtype=deeplake.types.Embedding(embedding_dim))
        perturbation_ds.add_column("label", dtype= deeplake.types.Int32)
        perturbation_ds.add_column("file_key", dtype= deeplake.types.Int32)

        print(f"{perturbation_type} strength: {strength:.4f}")
        network_handler.extract_embeddings(data_loader, perturbation_ds)

if __name__ == "__main__":
    main()