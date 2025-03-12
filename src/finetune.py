import os
from math import inf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import (
    get_args,
    save_args,
    get_dataset,
    log_metrics,
    Network,
    NetworkHandler
)

def main():
    arg_path = os.path.join("configs", "finetune.yaml")
    args = get_args(arg_path)
    num_workers = 0 if args["dataset"] == "pcam" else max(1, (os.cpu_count() // 4))

    data_dir = os.path.join("..", "embeddings" if args["embedding_mode"] else "data")
    encoder_dir = os.path.join("..", "assets", "model-weights", "pre-trained-weights")
    
    log_dir = os.path.join(
        "runs",
        args["dataset"], 
        args["encoder"], 
        "embedding-mode" if args["embedding_mode"] else "full-model-mode", 
        f"experiment-{args['experiment_num']}"
    )
    writer = SummaryWriter(log_dir)
    save_args(args, log_dir)
    
    model_dir = os.path.join(
        "..", 
        "assets", 
        "model-weights", 
        "finetune-weights", 
        args["dataset"], 
        args["encoder"], 
        "embedding-mode" if args["embedding_mode"] else "full-model-mode", 
        f"experiment-{args['experiment_num']}"
    )
    os.makedirs(model_dir, exist_ok=True)

    train_dataset = get_dataset(
        name=args["dataset"], 
        split="train", 
        data_dir=data_dir,
        embedding_mode=args["embedding_mode"],
        encoder=args["encoder"]
    )
    
    val_dataset = get_dataset(
        name=args["dataset"], 
        split="val" if args["dataset"] != "gleason-grading" else "test", 
        data_dir=data_dir,
        embedding_mode=args["embedding_mode"],
        encoder=args["encoder"]
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args["batch_size"], 
        shuffle=False,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        num_workers=num_workers
    )
    
    model = Network(args["encoder"], encoder_dir, args["num_classes"], args["freeze_encoder"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"], weight_decay=args["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    network_handler = NetworkHandler(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        precision=args["precision"],
        freeze_encoder=args["freeze_encoder"],
        embedding_mode=args["embedding_mode"]
    )   

    min_val_loss, max_val_accuracy = inf, -inf

    for epoch in range(1, args["epochs"] + 1):
        print(f"Epoch [{epoch}/{args['epochs']}]")
        
        train_loss, train_accuracy = network_handler.train_epoch(train_loader)
        log_metrics(writer=writer, loss=train_loss, prefix="Train", epoch=epoch, performance=train_accuracy)

        val_loss, val_accuracy = network_handler.validate_epoch(val_loader)
        log_metrics(writer=writer, loss=val_loss, prefix="Validation", epoch=epoch, performance=val_accuracy)

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