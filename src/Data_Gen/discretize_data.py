# File: discretize_data.py
# Description: Discretizes C-MAPSS sensor columns for DBN inference.

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def discretize_data(df, config=None, strategy='quantile', n_bins=4):
    """
    Discretizes sensor columns in the C-MAPSS dataset.

    Args:
        df (pd.DataFrame): Input dataframe with sensor columns (e.g., sensor_1 to sensor_21).
        config (dict): Optional dictionary of {'sensor_name': list of bin edges}.
        strategy (str): Binning strategy ('uniform', 'quantile', 'kmeans').
        n_bins (int): Number of bins to use for each sensor if config is not provided.

    Returns:
        pd.DataFrame: Original df with added discretized columns (sensor_x_disc).
    """
    df_discrete = df.copy()

    # Get only sensor columns
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    for col in sensor_cols:
        disc_col = f"{col}_disc"
        if config and col in config:
            # Manual binning from config
            bins = config[col]
            df_discrete[disc_col] = pd.cut(df_discrete[col], bins=bins, labels=False, include_lowest=True)
        else:
            # Auto binning using sklearn
            try:
                discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                reshaped = df_discrete[col].values.reshape(-1, 1)
                df_discrete[disc_col] = discretizer.fit_transform(reshaped).astype(int)
            except Exception as e:
                print(f"Warning: Skipping column {col} due to error: {e}")
                continue

    return df_discrete


# Example usage
if __name__ == "__main__":
    # Example fake data
    data = {
        'sensor_2': np.random.normal(600, 10, 100),
        'sensor_3': np.random.normal(1580, 15, 100),
    }
    df = pd.DataFrame(data)
    df_out = discretize_data(df, strategy='quantile', n_bins=3)
    print(df_out.head())
