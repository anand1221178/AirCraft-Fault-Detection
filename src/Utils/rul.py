# Utils/rul.py
import numpy as np
import pandas as pd

def compute_rul(df_probs: pd.DataFrame, fail_col="P(Engine_Core_Health=Fail)",
                thresh=0.8) -> pd.Series:
    """
    Returns a Series (same index) with cycles remaining until first Fail≥thresh.
    If the threshold is never reached ›∞ (np.inf).
    """
    fail_idx = np.where(df_probs[fail_col].values >= thresh)[0]
    if fail_idx.size == 0:
        # never hits the threshold – treat as 'infinite life'
        return pd.Series(np.inf, index=df_probs.index)
    T = fail_idx[0]
    steps_left = np.maximum(T - np.arange(len(df_probs)), 0)
    return pd.Series(steps_left, index=df_probs.index)
