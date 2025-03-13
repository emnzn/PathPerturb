import os
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parents[1])
RUN_DIR = os.path.join(SRC_DIR, "runs")
CONFIG_DIR = os.path.join(SRC_DIR, "configs")
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
ASSET_DIR = os.path.join(SRC_DIR, "..", "assets")
EMBEDDING_DIR = os.path.join(SRC_DIR, "..", "embeddings")