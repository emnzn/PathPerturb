import os

import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from timm.models.vision_transformer import VisionTransformer 

from utils import (
    get_args, 
    get_encoder,
    save_embeddings
    )

def get_embeddings(
    encoder: VisionTransformer,
    dataloader: DataLoader,
    device: str,
    save_dir: str
    ):

    results = {
        "image": [],
        "label": [],
        "embedding": []
    }
    
    encoder.eval()
    with torch.inference_mode():
        for img, label in tqdm(dataloader, desc="Embedding ROIs"):
            img = img.to(device)
            embedding = encoder(img).cpu().numpy()

            results["image"].extend(img.cpu().numpy())
            results["label"].extend(label.numpy())
            results["embedding"].extend(embedding)

    save_embeddings(results, save_dir)
    

def main():
    load_dotenv(os.path.join("..", ".env"))
    hf_token = os.getenv("HF_TOKEN")
    os.environ["HUGGINGFACE_TOKEN"] = hf_token

    arg_dir = os.path.join("configs", "embed.yaml")
    args = get_args(arg_dir)

    data_dir = os.path.join("..", "data")
    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    encoder_dir = os.path.join("..", "assets", "pre-trained-weights")
    dest_dir = os.path.join("..", "data", "embeddings", args["data_split"], args["encoder"])
    os.makedirs(dest_dir, exist_ok=True)

    dataset = PCAM(data_dir, split=args["data_split"], transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=args["batch_size"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(args["encoder"], encoder_dir, device)
    get_embeddings(encoder, dataloader, device, os.path.join(dest_dir, "base.parquet"))

    brightness_range = np.linspace(
        args["brightness_min"],
        args["brightness_max"],
        int((args["brightness_max"] - args["brightness_min"]) / 0.5)
    )

    contrast_range = np.linspace(
        args["contrast_min"],
        args["contrast_max"],
        int((args["contrast_max"] - args["contrast_min"]) / 0.5)
    )

    saturation_range = np.linspace(
        args["saturation_min"],
        args["saturation_max"],
        int((args["saturation_max"] - args["saturation_min"]) / 0.5)
    )

    hue_range = np.linspace(-0.5, 0.5, int((0.5 + 0.5) / 0.1))
    
    augmentation_array = {
        "brightness": brightness_range,
        "contrast": contrast_range,
        "saturation": saturation_range,
        "hue": hue_range
    }

    for aug in augmentation_array.keys():
        aug_range = augmentation_array[aug]

        for strength in aug_range:
            if aug == "brightness":
                transform = transforms.Compose([
                    transforms.ColorJitter(brightness=(strength, strength)),
                    transforms.ToTensor()
                ])

            if aug == "contrast":
                transform = transforms.Compose([
                    transforms.ColorJitter(contrast=(strength, strength)),
                    transforms.ToTensor()
                ])

            if aug == "saturation":
                transform = transforms.Compose([
                    transforms.ColorJitter(saturation=(strength, strength)),
                    transforms.ToTensor()
                ])

            if aug == "hue":
                transform = transforms.Compose([
                    transforms.ColorJitter(hue=(strength, strength)),
                    transforms.ToTensor()
                ])

            dataset = PCAM(data_dir, split=args["data_split"], transform=transform)
            dataloader = DataLoader(dataset, batch_size=args["batch_size"])
            get_embeddings(encoder, dataloader, device, os.path.join(dest_dir, f"{aug}-{strength}.parquet"))

if __name__ == "__main__":
    main()