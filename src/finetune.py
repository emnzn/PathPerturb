import os
from math import inf

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.datasets import PCAM
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score

from utils import (
    get_args,
    Network
    )

def train(
    dataloader,
    criterion,
    optimizer, 
    model,
    device
    ):

    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }
    
    model.train()
    for img, target in tqdm(dataloader, desc="Training in progress"):
        img = img.to(device)
        target =  target.to(device)

        logits, _ = model(img)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["running_loss"] += loss.detach().cpu().item()
        metrics["predictions"].extend(pred.cpu().numpy())
        metrics["targets"].extend(target.cpu().numpy())

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

    return epoch_loss, epoch_balanced_accuracy

@torch.no_grad()
def validate(
    dataloader,
    criterion,
    model,
    device
    ):
    
    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }

    model.eval()
    for img, target in tqdm(dataloader, desc="Validation in progress"):
        img = img.to(device)
        target = target.to(device)

        logits, _ = model(img)
        loss = criterion(logits, target)

        confidence = F.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        metrics["running_loss"] += loss.cpu().item()
        metrics["predictions"].extend(pred.cpu().numpy())
        metrics["targets"].extend(target.cpu().numpy())

    epoch_loss = metrics["running_loss"] / len(dataloader)
    epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

    return epoch_loss, epoch_balanced_accuracy


def main():
    arg_path = os.path.join("configs", "finetune.yaml")
    args = get_args(arg_path)

    data_dir = os.path.join("..", "data")
    encoder_dir = os.path.join("..", "assets", "pre-trained-weights")
    log_dir = os.path.join("runs", args["encoder"])
    
    model_dir = os.path.join("..", "assets", "finetune-weights", args["encoder"])
    os.makedirs(model_dir, exist_ok=True)

    base_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    writer = SummaryWriter(log_dir)
    train_dataset = PCAM(data_dir, split="train", transform=base_transform)
    val_dataset = PCAM(data_dir, split="val", transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Network(args["encoder"], encoder_dir, args["num_classes"], args["freeze_encoder"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    min_val_loss, max_val_accuracy = inf, -inf

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")
        
        train_loss, train_accuracy = train(train_loader, criterion, optimizer, model, device)
        
        print("Train Statistics:")
        print(f"Loss: {train_loss:.4f} | Balanced Accuracy: {train_accuracy:.4f}\n")

        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Balanced-Accuracy", train_accuracy, epoch)

        val_loss, val_accuracy = validate(val_loader, criterion, model, device)
        
        print("Validation Statistics:")
        print(f"Loss: {val_loss:.4f} | Balanced Accuracy: {val_accuracy:.4f}\n")

        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_accuracy, epoch)

        if val_loss < min_val_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, f"lowest-loss.pth"))
            min_val_loss = val_loss
            print("New minimum loss — model saved.")
        
        if val_accuracy > max_val_accuracy:
            torch.save(model.state_dict(), os.path.join(model_dir, f"highest-balanced-accuracy.pth"))
            max_val_accuracy = val_accuracy
            print("New maximum balanced accuracy — model saved.")
    
        scheduler.step()

        print("-------------------------------------------------------------------\n")

    print("Run Summary:")
    print(f"Min Loss: {min_val_loss:.4f} | Max Balanced Accuracy: {max_val_accuracy:.4f}\n")

        
if __name__ == "__main__":
    main()  