import os

import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked
from timm.models.vision_transformer import VisionTransformer 

class Network(nn.Module):
    """
    Initializes the network with the foundation model as the encoder
    and a linear layer as the classifier.

    Parameters
    ----------
    encoder: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.

    num_classes: int
        The number of classes to be classified.
    
    freeze_encoder: bool
        Whether to freeze the encoder during finetuning.
    """

    def __init__(
        self,
        encoder: str,
        encoder_dir: str,
        num_classes: int,
        freeze_encoder: bool = True
        ):
        super().__init__()

        self.encoder = get_encoder(encoder, encoder_dir, "cpu")
        self.fc = get_classification_head(encoder, num_classes)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.encoder.eval()

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.fc(embedding)

        return logits, embedding

def download_weights(
    encoder: str,
    encoder_dir: str
    ):
    """
    Downloads the weights of a foundation model to a selected directory.

    Parameters
    ----------
    encoder: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.
    """

    valid_encoders = [
        "uni",
        "gigapath",
        "virchow"
    ]

    assert encoder in valid_encoders, f"encoder must be one of {valid_encoders}"

    if encoder == "uni":
        encoder_path = os.path.join(encoder_dir, "uni.pth")
        encoder = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    if encoder == "gigapath":
        encoder_path = os.path.join(encoder_dir, "gigapath.pth")
        encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)

    if encoder == "virchow":
        encoder_path = os.path.join(encoder_dir, "virchow.pth")
        encoder = timm.create_model("hf-hub:paige-ai/Virchow", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    torch.save(encoder, encoder_path)


def get_encoder(
    encoder: str,
    encoder_dir: str,
    device: str
    ) -> VisionTransformer:
    """
    Returns an initialized foundation encoder.

    Parameters
    ----------
    encoder: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    encoder_dir: str
        The directory containing the encoder weights.

    device: str
        The device to map the weights onto.

    Returns
    -------
    encoder: VisionTransformer
        The initialized foundation encoder.
    """

    valid_encoders = [
        "uni",
        "gigapath",
        "virchow"
    ]

    assert encoder in valid_encoders, f"encoder must be one of {valid_encoders}"
    os.makedirs(encoder_dir, exist_ok=True)

    if encoder == "uni":
        encoder_path = os.path.join(encoder_dir, "uni.pth")
    
    if encoder == "gigapath":
        encoder_path = os.path.join(encoder_dir, "gigapath.pth")

    if encoder == "virchow":
        encoder_path = os.path.join(encoder_dir, "virchow.pth")

    if not os.path.isfile(encoder_path):
        download_weights(encoder, encoder_dir=encoder_dir)

    encoder = torch.load(encoder_path, map_location=torch.device(device), weights_only=False)

    return encoder

class ClassificationHead(nn.Module):
    """
    Initializes a linear classification head.

    Parameters
    ----------
    in_dim: int
        The input dimension of the classifier.

    out_dim: int
        The output dimension of the classifier or 
        the number of classes.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logits = self.head(x)

        return logits
    
def get_classification_head(
    encoder: str,
    num_classes: int
    ) -> ClassificationHead:
    """
    Initializes the appropriate classification head
    according to a selected foundation encoder.

    Parameters
    ----------
    encoder: str
        The foundation model to be used as the encoder.
        One of [uni, gigapath, virchow].

    num_classes: int
        The number of output classes.
    """


    valid_encoders = [
        "uni",
        "gigapath",
        "virchow"
    ]

    assert encoder in valid_encoders, f"encoder must be one of {valid_encoders}"

    if encoder == "uni":
        head = ClassificationHead(in_dim=1024, out_dim=num_classes)

    if encoder == "gigapath":
        head = ClassificationHead(in_dim=1536, out_dim=num_classes)

    if encoder == "virchow":
        head = ClassificationHead(in_dim=1280, out_dim=num_classes)

    return head