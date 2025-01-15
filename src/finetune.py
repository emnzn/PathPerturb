import os

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.funnctional as F
from torch.utils.tensorboard import SummaryWriter

from utils import (
    get_args,
    FinetuneDataset
    )

def train(
    dataloader,
    criterion,
    optimizer, 
    model,
    device
    ):
    pass

def validate(
    dataloader,
    criterion,
    model,
    device
    ):
    pass

def main():
    pass

if __name__ == "__main__":
    main()