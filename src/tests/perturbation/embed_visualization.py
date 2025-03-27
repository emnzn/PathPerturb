import os
import sys
import json
from typing import Dict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import deeplake
import numpy as np
import pandas as pd
from umap import UMAP
from tqdm import tqdm
import plotly.express as px

from utils.constants import (
    DATA_DIR,
    CONFIG_DIR,
    RESULTS_DIR,
    EMBEDDING_DIR
)

from utils import set_seed, get_args

def subsample(
    ds: deeplake.Dataset, 
    sample_percantage: float,
    label_map: Dict[int, str]
    ) -> pd.DataFrame:

    if sample_percantage < 0 or sample_percantage > 1:
        raise ValueError(f"sample_percantage must be between 0 and 1")
    
    unique_labels = np.unique(ds["label"][:]).tolist()

    embeddings = []
    labels = []

    for label in tqdm(unique_labels, desc="Processing Embeddings"):
        filtered = ds.query(f"SELECT * WHERE label == {label}")
        indices = np.random.randint(
            low=0,
            high=len(filtered)-1,
            size=int(len(filtered)*sample_percantage)
        )

        sampled = filtered[*indices]
        embeddings.extend(sampled["embedding"])
        labels.extend(sampled["label"])

    df = pd.DataFrame({
        "embedding": embeddings,
        "label": labels
    })

    df["label"] = df["label"].map(lambda x: label_map[x])

    return df

def generate_projection(
    df: pd.DataFrame, 
    dest_dir: str, 
    save_filename: str,
    height: int = 800,
    width: int = 800
    ):

    umap_2d = UMAP()
    proj_2d = umap_2d.fit_transform(X=np.stack(df["embedding"].tolist(), axis=0))

    fig_2d = px.scatter(
        proj_2d,
        x=0,
        y=1,
        color=df["label"],
        labels={"color": "label"},
        height=height,
        width=width
    )

    fig_2d.write_image(os.path.join(dest_dir, save_filename))

def main():
    arg_path = os.path.join(CONFIG_DIR, "embed_visualization.yaml")
    args = get_args(arg_path)
    set_seed(args["seed"])
    
    label_map_path = os.path.join(DATA_DIR, args["dataset"], "label-map.json")
    
    with open(label_map_path, "r") as f: 
        label_map = json.load(f)
    
    label_map = {v: k for k, v in label_map.items()}

    perturbation_dir = os.path.join(EMBEDDING_DIR, args["dataset"], args["encoder"], "perturbations", args["perturbation_type"])
    perturbed_datasets = sorted(os.listdir(perturbation_dir), key=lambda x: float(x.split("_")[-1]))

    dest_dir = os.path.join(RESULTS_DIR, args["dataset"], args["encoder"], "perturbations", args["perturbation_type"], "separability")
    os.makedirs(dest_dir, exist_ok=True)

    for ds in perturbed_datasets:
        ds_path = os.path.join(perturbation_dir, ds)
        augmentation_strength = ds_path.split("_")[-1]

        print(f"Visualizing embeddings for {args['perturbation_type']} at {augmentation_strength} intensity:")

        embedding_dataset = deeplake.open_read_only(ds_path)
        df = subsample(embedding_dataset, sample_percantage=args["sample_percentage"], label_map=label_map)

        generate_projection(
            df=df,
            dest_dir=dest_dir,
            save_filename=f"{ds}.png",
            height=args["height"],
            width=args["width"]
        )
        print("\n-------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
