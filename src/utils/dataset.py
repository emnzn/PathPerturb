import torch
import pandas as pd
import torch.nn as nn

class FinetuneDataset(nn.Module):
    def __init__(
        self, 
        table_path: str
        ):
        super().__init__()
        self.df = pd.read_parquet(table_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embedding = torch.tensor(row["embedding"])
        target = row["label"]

        return embedding, target