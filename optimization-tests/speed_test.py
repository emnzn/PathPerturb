import os
import time

import torch
import deeplake
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

def transform_fn(row):
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    img = img_transform(row["image"])
    label = torch.tensor(row["label"])
    file_key = torch.tensor(row["file_key"])

    return img, label, file_key

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
    data_dir = os.path.join("..", "data", "gleason-grading")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    print("Speed testing train set:")
    train_ds = init_speedtest(train_dir)
    train_loader = DataLoader(
        train_ds.pytorch(transform=transform_fn),
        batch_size=8,
        shuffle=True
        )
    
    print("Train set Loading:")
    load_speedtest(train_loader)

    print("Speed testing test set:")
    test_ds = init_speedtest(test_dir)
    test_loader = DataLoader(
        test_ds.pytorch(transform=transform_fn),
        batch_size=8,
        shuffle=False
        )
    
    print("Test set Loading:")
    load_speedtest(test_loader)
    
if __name__ == "__main__":
    main()