import os

import timm
import torch
import torch.nn as nn
from timm.layers import SwiGLUPacked


class Network(nn.Module):
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
    ):

    valid_encoders = [
        "uni",
        "gigapath",
        "virchow"
    ]

    assert encoder in valid_encoders, f"encoder must be one of {valid_encoders}"

    if encoder == "uni":
        encoder_path = os.path.join(encoder_dir, "uni.pth")
    
    if encoder == "gigapath":
        encoder_path = os.path.join(encoder_dir, "gigapath.pth")

    if encoder == "virchow":
        encoder_path = os.path.join(encoder_dir, "virchow.pth")

    if not os.path.isfile(encoder_path):
        download_weights(encoder)

    encoder = torch.load(encoder_path, map_location=torch.device(device))

    return encoder

class ClassificationHead(nn.Module):
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