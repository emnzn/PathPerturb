from typing import Tuple, Literal

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from deeplake import Dataset as DeepLakeDataset
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import balanced_accuracy_score

from .network import Network

class NetworkHandler:
    """
    This class encapsulates all computation logic for a given neural network,
    including training, validation, inference and embedding exxtraction.
    
    Supports mixed precision training.

    Attributes
    ----------
    model: Network
        The neural network.

    criterion: nn.Module
        The function for loss computation.

    optimizer: torch.optim.Optimizer
        The optimizer for gradient descent.

    scheduler: LRScheduler
        The learning rate scheduler.

    freeze_encoder: bool
        Whether the encoder is frozen.
        Will be used as a flag in switching between train and eval modes.

    device: str
        The device for computations.

    use_amp: bool
        Whether to use mixed precision training.

    grad_scaler: GradScaler
        Scaler for mixed precision training.

    embedding_mode: bool
        Whether to perform computations on pre-extracted embeddings.
    """

    def __init__(
        self,
        model: Network,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None, 
        scheduler: LRScheduler = None,
        precision: Literal["full", "mixed"] = "full",
        freeze_encoder: bool = True,
        embedding_mode: bool = False
        ):  

        """
        Parameters
        ----------
        model: Network
            The neural network.

        criterion: nn.Module
            The function for loss computation.

        optimizer: torch.optim.Optimizer
            The optimizer for gradient descent.

        scheduler: LRScheduler
            The learning rate scheduler.

        precision: Literal["full", "mixed"]
            Whether to train in mixed or full precision.
            Must be one of ['full', 'mixed'].

        freeze_encoder: bool
            Whether the encoder is frozen.
            Will be used as a flag in switching between train and eval modes.

        embedding_mode: bool
            Whether to perform computations on pre-extracted embeddings.
        """

        valid_precisions = ["full", "mixed"]

        if precision not in valid_precisions:
            raise ValueError(f"precision must be one of  {valid_precisions}.")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.freeze_encoder = freeze_encoder
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = precision == "mixed" and self.device == "cuda"
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        self.model = self.model.to(self.device)
        self.embedding_mode = embedding_mode

        if self.device != "cuda" and precision == "mixed":
            raise ValueError(f"Mixed precision unavailable with current device: {self.device}. Switch to full precision.\n")
            
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Trains the model for 1 epoch.

        Parameters
        ----------
        train_loader: DataLoader
            The data loader for training.

        Returns
        -------
        epoch_loss: float
            The loss for the epoch.

        epoch_balanced_accuracy: float
            The average balanced accuracy for the given epoch.  
        """
        
        metrics = {
            "running_loss": 0,
            "predictions": [],
            "targets": []
        }
        
        self.model.train()
        if self.freeze_encoder or self.embedding_mode: self.model.encoder.eval()

        pbar = tqdm(train_loader, desc="Training in progress")

        for patch, target, *_ in pbar:
            patch = patch.to(self.device)
            target =  target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch) 
                loss = self.criterion(logits, target)

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()

            confidence = F.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["running_loss"] += loss.detach().cpu().item()
            metrics["predictions"].extend(pred.cpu().numpy())
            metrics["targets"].extend(target.cpu().numpy())

            pbar.set_postfix({"loss": loss.detach().cpu().item()})

        epoch_loss = metrics["running_loss"] / len(train_loader)
        epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

        return epoch_loss, epoch_balanced_accuracy
    
    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Runs validation for 1 epoch.

        Parameters
        ----------
        val_loader: DataLoader
            The data loader for validation.

        Returns
        -------
        epoch_loss: float
            The loss for the epoch.

        epoch_balanced_accuracy: float
            The average balanced accuracy for the given epoch.  
        """

        metrics = {
            "running_loss": 0,
            "predictions": [],
            "targets": []
        }

        self.model.eval()
        pbar = tqdm(val_loader, desc="Validation in progress")
        for patch, target, *_ in pbar:
            patch = patch.to(self.device)
            target = target.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model.fc(patch) if self.embedding_mode else self.model(patch)
                loss = self.criterion(logits, target)

            confidence = torch.softmax(logits, dim=1)
            pred = torch.argmax(confidence, dim=1)

            metrics["running_loss"] += loss.detach().cpu().item()
            metrics["predictions"].extend(pred.cpu().numpy())
            metrics["targets"].extend(target.cpu().numpy())

            pbar.set_postfix({"loss": loss.detach().cpu().item()})

        epoch_loss = metrics["running_loss"] / len(val_loader)
        epoch_balanced_accuracy = balanced_accuracy_score(metrics["targets"], metrics["predictions"])

        return epoch_loss, epoch_balanced_accuracy
    
    @torch.no_grad()
    def extract_embeddings(
        self, 
        embed_loader: DataLoader,
        deeplake_ds: DeepLakeDataset
        ):
        
        self.model.eval()
        pbar = tqdm(embed_loader, desc="Generating embeddings")
        for patch, label, file_key in pbar:
            patch = patch.to(self.device)
            embedding = self.model.encoder(patch).detach().cpu()
            
            deeplake_ds.append({
                "embedding": embedding.numpy(),
                "label": label.numpy(),
                "file_key": file_key.numpy()
            })