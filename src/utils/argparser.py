import os
import yaml
from typing import Dict, Union

import numpy as np

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

def save_args(
    args: Dict[str, Union[float, str]], 
    dest_dir: str
    ):
    """
    Saves the arguments as a yaml file in a given destination directory.
    
    Parameters
    ----------
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.

    dest_dir: str
        The the directory of the save destination.
    """

    path = os.path.join(dest_dir, "run-config.yaml")
    with open(path, "w") as f:
        yaml.dump(args, f)

def get_transform_range(args: Dict[str, Union[float, str]]) -> np.ndarray:
    """
    Gets the range of transformation intensities to iterate over.

    Parameters
    ----------
    args: Dict[str, Union[float, str]]
        The inference arguments.

    Returns
    -------
    transform_range: np.ndarray
        The range of transformation intensities to be applied to the dataset.
    """

    step = args["augmentation_step"]
    maximum = args["augmentation_max"]
    minimum = args["augmentation_min"]

    transform_range = np.linspace(
        minimum,
        maximum,
        int((maximum - minimum) / step)
    )

    return transform_range