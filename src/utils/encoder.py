import os

import timm
import torch
from timm.layers import SwiGLUPacked

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
