from torch.utils.tensorboard import SummaryWriter

def log_metrics(
    writer: SummaryWriter, 
    loss: float,
    prefix: str,
    epoch: int,
    performance: float
    ):

    """
    Logging function.
    """

    print(f"{prefix} Statistics:")
    writer.add_scalar(f"{prefix}/Loss", loss, epoch)

    print(f"Loss: {loss:.4f} | Balanced Accuracy: {performance:.4f}\n")
    writer.add_scalar(f"{prefix}/Balanced-Accuracy", performance, epoch)
    