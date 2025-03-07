import os
from math import inf
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score

from utils import (
    get_args,
    save_args,
    get_dataset,
    log_metrics,
    Network
    )

def train(
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer, 
    model: Network,
    device: str,
    use_amp: bool
    ) -> Tuple[float, float]:

    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader: DataLoader
        The data loader to iterate over.

    criterion: nn.Module
        The loss function.

    optimizer: optim.Optimizer
        The optimizer for parameter updates.

    model: Network
        The model to be trained.

    device: str
        One of [cuda, cpu].

    use

    Returns
    -------
    epoch_loss: float
        The average loss for the given epoch.

    epoch_balanced_accuracy: float
        The average balanced accuracy for the given epoch.  
    """

    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }
    
    model.train()
    for img, target, *_ in tqdm(dataloader, desc="Training in progress"):
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
    dataloader: DataLoader,
    criterion: nn.Module,
    model: Network,
    device: str
    ):

    """
    Runs validation for a single epoch.
    """
    
    metrics = {
        "running_loss": 0,
        "predictions": [],
        "targets": []
    }

    model.eval()
    for img, target, *_ in tqdm(dataloader, desc="Validation in progress"):
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
    encoder_dir = os.path.join("..", "assets", "model-weights", "pre-trained-weights")
    
    log_dir = os.path.join("runs", args["dataset"], args["encoder"], f"experiment-{args['experiment_num']}")
    writer = SummaryWriter(log_dir)
    save_args(args, log_dir)
    
    model_dir = os.path.join("..", "assets", "model-weights", "finetune-weights", args["dataset"], args["encoder"], f"experiment-{args['experiment_num']}")
    os.makedirs(model_dir, exist_ok=True)

    train_dataset = get_dataset(
        name=args["dataset"], 
        split="train", 
        data_dir=data_dir
        )
    
    val_dataset = get_dataset(
        name=args["dataset"], 
        split="val" if args["dataset"] != "gleason-grading" else "test", 
        data_dir=data_dir
        )

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    model = Network(args["encoder"], encoder_dir, args["num_classes"], args["freeze_encoder"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    min_val_loss, max_val_accuracy = inf, -inf

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")
        
        train_loss, train_accuracy = train(
            dataloader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            model=model, 
            device=device,
            use_amp=use_amp
            )
        
        log_metrics(
            writer=writer,
            loss=train_loss,
            prefix="Train",
            epoch=epoch,
            performance=train_accuracy
            )

        val_loss, val_accuracy = validate(
            dataloader=val_loader, 
            criterion=criterion, 
            model=model, 
            device=device
            )
        
        log_metrics(
            writer=writer,
            loss=val_loss,
            prefix="Validation",
            epoch=epoch,
            performance=val_accuracy
            )

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