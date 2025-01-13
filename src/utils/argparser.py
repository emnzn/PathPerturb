import os
import yaml
from typing import Dict, Union

def get_args(arg_dir: str) -> Dict[str, Union[float, str]]:
    """
    Gets relevant arguments from a yaml file.

    Parameters
    ----------
    arg_dir: str
        The path to the yaml file containing the arguments.
    
    Returns
    -------    
    args: Dict[str, Union[float, str]]
        The arguments in the form of a dictionary.
    """
    
    with open(arg_dir, "r") as f:
        args = yaml.safe_load(f)

    return args