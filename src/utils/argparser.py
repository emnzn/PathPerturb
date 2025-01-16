import yaml
from typing import Dict, Union

import numpy as np
import torchvision.transforms as transforms

def get_args(arg_path: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    arg_path: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_path, "r") as f:
        args = yaml.safe_load(f)

    return args

def get_transform_range(args: Dict[str, Union[float, str]]):
    step = args["augmentation_step"]
    maximum = args["augmentation_max"]
    minimum = args["augmentation_min"]

    transform_range = np.linspace(
        minimum,
        maximum,
        int((maximum - minimum) / step)
    )

    return transform_range