import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Any
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import deeplake
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.constants import (
    RUN_DIR,
    CONFIG_DIR,
    RESULTS_DIR,
    EMBEDDING_DIR,
    BASE_MODEL_DIR
)

from utils import (
    get_args,
    Network,
    NetworkHandler
)

def transform_fn(row: Dict[str, Any]) -> Tuple[torch.Tensor]:
    embedding = torch.tensor(row["embedding"])
    label = torch.tensor(row["label"], dtype=torch.long)
    file_key = torch.tensor(row["file_key"], dtype=torch.long)

    return embedding, label, file_key

def main():
    arg_path = os.path.join(CONFIG_DIR, "inference_perturbation.yaml")
    args = get_args(arg_path)
    num_workers = 0 if args["dataset"] == "pcam" else max(1, (os.cpu_count() // 4))

    run_path = os.path.join(
        RUN_DIR,
        args["dataset"], 
        args["encoder"], 
        "embedding-mode", 
        f"experiment-{args['experiment_num']}",
        "run-config.yaml"
    )
    run_config = get_args(run_path)

    encoder_dir = os.path.join(BASE_MODEL_DIR, "pre-trained-weights")
    weight_path = os.path.join(
        BASE_MODEL_DIR, 
        "finetune-weights", 
        args["dataset"], 
        args["encoder"], 
        "embedding-mode", 
        f"experiment-{args['experiment_num']}",
        "highest-balanced-accuracy.pth"
    )
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)

    criterion = nn.CrossEntropyLoss(reduction="none")
    model = Network(args["encoder"], encoder_dir, num_classes=run_config["num_classes"])
    model.load_state_dict(weights)
    
    network_handler = NetworkHandler(
        model=model,
        criterion=criterion,
        precision=args["precision"],
        embedding_mode=True
    )

    save_dir = os.path.join(RESULTS_DIR, args["dataset"], args["encoder"], "perturbations", args["perturbation_type"])
    os.makedirs(save_dir, exist_ok=True)

    perturbation_dir = os.path.join(EMBEDDING_DIR, args["dataset"], args["encoder"], "perturbations", args["perturbation_type"])
    perturbation_datasets = sorted(os.listdir(perturbation_dir), key=lambda x: float(x.split("_")[-1]), reverse=False)

    for ds in perturbation_datasets:
        augmentation_strength = ds.split("_")[-1]
        ds_path = os.path.join(perturbation_dir, ds)

        inference_dataset = deeplake.open_read_only(ds_path).pytorch(transform=transform_fn)
        inference_loader = DataLoader(inference_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=num_workers)

        print(f"Evaluating {args['perturbation_type']} at {augmentation_strength}% intensity:")
        iteration_loss, iteration_balanced_accuracy = network_handler.inference(inference_loader, save_dir=save_dir, save_filename=ds)
        
        print(f"Loss: {iteration_loss:.4f} | Balanced Accuracy: {iteration_balanced_accuracy:.4f}\n")
        print("-------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()