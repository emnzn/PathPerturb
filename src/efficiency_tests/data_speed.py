import os
import sys
import time
sys.path.append("../")

import deeplake
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import get_args, transform_fn

def init_speedtest(data_dir):
    start = time.time()
    dataset = deeplake.open(data_dir)
    total_time = time.time() - start

    print(f"Dataset initialization speed: {total_time:.4f} seconds\n")

    return dataset

def load_speedtest(loader):
    start = time.time()
    
    for _ in tqdm(loader, desc="Speed test running"):
        continue

    total_time = time.time() - start 
    return total_time
    
def main():
    arg_dir = os.path.join("..", "configs", "speed-test.yaml")
    args = get_args(arg_dir)

    data_dir = os.path.join("..", "..", "data", args["dataset"])
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    print("Speed testing train set:")
    train_ds = init_speedtest(train_dir)
    train_loader = DataLoader(
        train_ds.pytorch(transform=transform_fn),
        batch_size=args["batch_size"],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        num_workers=(os.cpu_count() // 4)
        )
    
    print("Train set Loading:")
    load_speedtest(train_loader)

    print("Speed testing test set:")
    test_ds = init_speedtest(test_dir)
    test_loader = DataLoader(
        test_ds.pytorch(transform=transform_fn),
        batch_size=args["batch_size"],
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
        num_workers=(os.cpu_count() // 4)
        )
    
    print("Test set Loading:")
    load_speedtest(test_loader)
    
if __name__ == "__main__":
    main()