from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class FoldConfig:
    df: pd.DataFrame
    folds_train: List
    folds_val: List
    folds_test: List
    folds_pred: List
    label_df: Optional[pd.DataFrame] = None
