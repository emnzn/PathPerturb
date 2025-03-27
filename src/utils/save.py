import os
from typing import Dict, Union

import numpy as np
import pandas as pd

def save_inference_table(
    data_dict: Dict[str, Union[str, np.ndarray]],
    save_dir: str,
    save_filename: str
    ) -> None:

    """
    Saves inference results into a csv file for downstream analysis.
    """

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(data_dict)
    df.to_parquet(os.path.join(save_dir, f"{save_filename}.parquet"), index=False)