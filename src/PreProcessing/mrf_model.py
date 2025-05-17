# File: mrf_model.py
# Description: Vibration sensor smoothing using temporal or pairwise MRF logic

import pandas as pd
import numpy as np

def temporal_mrf_smoothing(series, window=3):
    """
    Applies temporal smoothing to a discrete sensor sequence.
    Penalizes rapid state changes over a moving window.

    Args:
        series (pd.Series): Discrete sensor readings (e.g., 0–3).
        window (int): Size of the smoothing window.

    Returns:
        pd.Series: Smoothed sequence.
    """
    smoothed = series.copy()
    for i in range(1, len(series) - 1):
        window_vals = series[max(i - window, 0):min(i + window + 1, len(series))]
        most_common = window_vals.mode()
        if not most_common.empty:
            smoothed.iloc[i] = most_common[0]
    return smoothed

def pairwise_mrf_smoothing(df, sensor1, sensor2):
    """
    Applies MRF smoothing to two colocated vibration sensors.
    Forces consistency using a simple pairwise rule.

    Args:
        df (pd.DataFrame): DataFrame with two sensor columns.
        sensor1 (str): Name of first vibration sensor column.
        sensor2 (str): Name of second vibration sensor column.

    Returns:
        pd.DataFrame: Copy of df with smoothed sensor columns.
    """
    df = df.copy()
    for i in range(len(df)):
        v1 = df.at[i, sensor1]
        v2 = df.at[i, sensor2]
        if pd.notnull(v1) and pd.notnull(v2) and v1 != v2:
            df.at[i, sensor1] = df.at[i, sensor2] = int(round((v1 + v2) / 2))
    return df

# Example usage
if __name__ == "__main__":
    test_data = pd.DataFrame({
        'sensor_4_disc': [0, 0, 3, 0, 0, 2, 0, 1, 0, 0],
        'sensor_11_disc': [0, 1, 3, 0, 0, 2, 0, 1, 1, 0]
    })
    smoothed_df = pairwise_mrf_smoothing(test_data, 'sensor_4_disc', 'sensor_11_disc')
    smoothed_df['sensor_4_smoothed'] = temporal_mrf_smoothing(smoothed_df['sensor_4_disc'])
    print(smoothed_df)