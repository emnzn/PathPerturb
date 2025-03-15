import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import deeplake
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import get_args, img_transform_fn
from utils.constants import DATA_DIR, CONFIG_DIR

def init_speedtest(data_dir):
    """
    Initializes a dataset whilst timing its speed of initialization.
    """
    start = time.time()
    dataset = deeplake.open(data_dir)
    total_time = time.time() - start

    print(f"Dataset initialization speed: {total_time:.4f} seconds\n")

    return dataset

def load_speedtest(loader):
    """
    Iterates through the entire dataset through a data loader and
    timing the speed of the full iteration.
    """
    start = time.time()
    
    for _ in tqdm(loader, desc="Speed test running"):
        continue

    total_time = time.time() - start 
    return total_time
    
def main():
    arg_dir = os.path.join(CONFIG_DIR, "data_speed.yaml")
    args = get_args(arg_dir)

    data_dir = os.path.join(DATA_DIR, args["dataset"])
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    print("Speed testing train set:")
    train_ds = init_speedtest(train_dir)
    train_loader = DataLoader(
        train_ds.pytorch(transform=img_transform_fn),
        batch_size=args["batch_size"],
        shuffle=True
    )
    
    print("Train set Loading:")
    load_speedtest(train_loader)

    print("Speed testing test set:")
    test_ds = init_speedtest(test_dir)
    test_loader = DataLoader(
        test_ds.pytorch(transform=img_transform_fn),
        batch_size=args["batch_size"],
        shuffle=False
    )
    
    print("Test set Loading:")
    load_speedtest(test_loader)
    
if __name__ == "__main__":
    main()