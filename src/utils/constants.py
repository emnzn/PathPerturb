import os
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parents[2])
SRC_DIR = os.path.join(ROOT_DIR, "src")
RUN_DIR = os.path.join(SRC_DIR, "runs")
CONFIG_DIR = os.path.join(SRC_DIR, "configs")

DATA_DIR = os.path.join(ROOT_DIR, "data")
ASSET_DIR = os.path.join(ROOT_DIR, "assets")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
EMBEDDING_DIR = os.path.join(ROOT_DIR, "embeddings")
BASE_MODEL_DIR = os.path.join(ROOT_DIR, "model-weights")


if __name__ == "__main__":
    dirs = {
        "ROOT_DIR": ROOT_DIR,
        "SRC_DIR": SRC_DIR,
        "RUN_DIR": RUN_DIR,
        "CONFIG_DIR": CONFIG_DIR,
        "DATA_DIR": DATA_DIR,
        "ASSET_DIR": ASSET_DIR,
        "BASE_MODEL_DIR": BASE_MODEL_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "EMBEDDING_DIR": EMBEDDING_DIR,
    }

    for name, path in dirs.items():
        print(f"{name}: {path}")
        print()