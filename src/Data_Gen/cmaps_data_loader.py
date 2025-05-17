# File: cmaps_data_loader.py
# Description: Loads and formats C-MAPSS data for integration into your PGM pipeline

import pandas as pd
import os
from config import HEALTH_NODE 

def load_cmaps_data(base_dir, dataset_id="FD001"):
    """
    Loads the train and test data for the selected C-MAPSS dataset.
    Automatically applies column names and returns merged DataFrames.

    Args:
        base_dir (str): Path to folder containing C-MAPSS txt files.
        dataset_id (str): One of "FD001", "FD002", "FD003", "FD004".

    Returns:
        train_df (pd.DataFrame): Train data with labeled RUL.
        test_df (pd.DataFrame): Test data.
        test_rul (pd.Series): Ground truth RUL for the test data (per unit).
    """
    assert dataset_id in ["FD001", "FD002", "FD003", "FD004"], f"Invalid dataset ID: {dataset_id}"

    # # File paths
    # train_file = os.path.join(base_dir, f"train_{dataset_id}.txt")
    # test_file = os.path.join(base_dir, f"test_{dataset_id}.txt")
    # rul_file = os.path.join(base_dir, f"RUL_{dataset_id}.txt")

    # Column names
    op_cols = [f"op_setting_{i}" for i in range(1, 4)]
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    col_names = ["unit", "cycle"] + op_cols + sensor_cols

    #files
    train_file = os.path.join(base_dir, f"train_{dataset_id}.txt")
    test_file = os.path.join(base_dir, f"test_{dataset_id}.txt")
    rul_file = os.path.join(base_dir, f"RUL_{dataset_id}.txt")

    # Load data
    train_df = pd.read_csv(train_file, sep="\s+", header=None, names=col_names)
    test_df = pd.read_csv(test_file, sep="\s+", header=None, names=col_names)
    test_rul = pd.read_csv(rul_file, sep="\s+", header=None)[0]

    return train_df, test_df, test_rul

def add_rul_column(df, max_rul_cap=None):
    df_copy = df.copy()
    max_cycles = df_copy.groupby("unit")["cycle"].max()
    df_copy = df_copy.merge(max_cycles.rename("max_cycle"), on="unit")
    df_copy["RUL"] = df_copy["max_cycle"] - df_copy["cycle"]
    if max_rul_cap is not None:
        df_copy["RUL"] = df_copy["RUL"].clip(upper=max_rul_cap)
    df_copy.drop("max_cycle", axis=1, inplace=True)
    return df_copy

def add_discrete_health_label(df):
    df_copy = df.copy() # Work on a copy
    def label_rul(rul): # These thresholds should ideally come from config.RUL_THRESHOLDS
        if rul > 120: # Example: config.RUL_THRESHOLDS["Healthy"]
            return 'Healthy'
        elif rul > 60: # Example: config.RUL_THRESHOLDS["Degrading"]
            return 'Degrading'
        else:
            return 'Critical'
    # Use the imported HEALTH_NODE variable (e.g., "Engine_Core_Health") for the column name
    df_copy[HEALTH_NODE] = df_copy["RUL"].apply(label_rul)
    return df_copy