import os
from typing import Dict, Union

import numpy as np
import pandas as pd

def save_parquet(
    data_dict: Dict[str, Union[str, np.ndarray]],
    dest_dir: str,
    split: str
    ) -> None:

    """
    Saves the embeddings into a parquet file for downstream experiments.
    """

    df = pd.DataFrame(data_dict)
    df.to_parquet(os.path.join(dest_dir, f"{split}.parquet"), index=False)