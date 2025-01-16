import os
from pathlib import Path

import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score

from utils import (
    get_args,
    Network,
    save_results,
    get_transform_range
    )

@torch.no_grad()
def inference(
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Network,
    device: str,
    save_dir: str
    ):

    """
    Runs inference on a given model.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    criterion: nn.Module
        The loss function.

    model: Network
        The model to be trained.

    device: str
        One of [cuda, cpu].

    save_dir: str
        The directory to save the results.
    """
    
    metrics = {
        "image": [],
        "target": [],
        "embedding": [],
        "prediction": [],
        "loss": [],
    }

    model.eval()
    for img, target in tqdm(dataloader, desc="Inference in progress"):
        img = img.to(device)
        target = target.to(device)

        logits, embedding = model(img)
        loss = criterion(logits, target)

        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["image"].extend(img.cpu().numpy())
        metrics["target"].extend(target.cpu().numpy())
        metrics["embedding"].extend(embedding.cpu().numpy())
        metrics["prediction"].extend(pred.cpu().numpy())
        metrics["loss"].extend(loss.cpu().numpy())

    dataset_loss = sum(metrics["loss"]) / len(dataloader)
    dataset_balanced_accuracy = balanced_accuracy_score(metrics["target"], metrics["prediction"])
    save_results(metrics, save_dir)

    print(f"Loss: {dataset_loss:.4f} | Balanced Accuracy: {dataset_balanced_accuracy:.4f}\n")
    print("-------------------------------------------------------------------\n")


def main():
    arg_path = os.path.join("configs", "inference.yaml")
    args = get_args(arg_path)
    data_dir = os.path.join("..", "data")
    dest_dir = os.path.join("..", "assets", "inference-tables", args["encoder"])
    os.makedirs(dest_dir, exist_ok=True)

    encoder_dir = os.path.join("..", "assets", "pre-trained-weights")
    model_dir = os.path.join("..", "assets", "finetune-weights", args["encoder"])

    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    inference_dataset = PCAM(data_dir, split="test", transform=base_transform)
    inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Network(args["encoder"], encoder_dir, args["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(reduction="none")

    state_dict = torch.load(
        os.path.join(model_dir, "highest-balanced-accuracy.pth"), 
        map_location=torch.device(device),
        weights_only=True
        )
    
    model.load_state_dict(state_dict)

    print("Baseline Dataset")

    inference(
        inference_loader, 
        criterion, 
        model, 
        device, 
        os.path.join(dest_dir, "base.parquet")
    )

    transform_range = get_transform_range(args)

    for strength in transform_range:
        if args["augmentation_mode"] == "brightness":
            transform = transforms.Compose([
                        transforms.ColorJitter(brightness=(strength, strength)),
                        transforms.ToTensor()
                    ])
            
        if args["augmentation_mode"] == "contrast":
            transform = transforms.Compose([
                        transforms.ColorJitter(contrast=(strength, strength)),
                        transforms.ToTensor()
                    ]) 
            
        if args["augmentation_mode"] == "saturation":
            transform = transforms.Compose([
                        transforms.ColorJitter(saturation=(strength, strength)),
                        transforms.ToTensor()
                    ]) 

        inference_dataset = PCAM(data_dir, split="test", transform=transform)
        inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False)
        
        print(f"{args['augmentation_mode']}-{strength:.4f}")

        inference(
            inference_loader, 
            criterion, 
            model, 
            device, 
            os.path.join(dest_dir, f"{args['augmentation_mode']}-{strength:.4f}.parquet")
        )


if __name__ == "__main__":
    main()