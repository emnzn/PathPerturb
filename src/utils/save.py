from typing import Dict, Union

import numpy as np
import pandas as pd

def save_results(
    results: Dict[str, Union[str, np.ndarray]],
    save_dir: int
    ) -> None:

    """
    Saves the images, labels and embeddings from the dataset.

    Parameters
    ----------
    results: Dict[str, Union[str, np.ndarray]]
        A dictionary containing the images, labels and embeddings from the dataset.
    """

    df = pd.DataFrame(results)
    df["image"] = df["image"].map(lambda x: x.flatten())
    df.to_parquet(save_dir, index=False)