# File: baselines.py
# Description: Rule-based classifier for C-MAPSS sensor data

import pandas as pd

def rule_based_health_label(df):
    """
    Applies rule-based logic to estimate health states from raw sensor readings.
    You can adjust thresholds based on sensor behavior observed in C-MAPSS.

    Args:
        df (pd.DataFrame): DataFrame containing sensor columns.

    Returns:
        pd.Series: Predicted health states as strings ('Healthy', 'Degrading', 'Critical')
    """
    def classify_row(row):
        # Example: sensor_4 (vibration), sensor_2 (N2), sensor_3 (EGT)
        vib = row.get("sensor_4", None)
        egt = row.get("sensor_3", None)
        n2 = row.get("sensor_2", None)

        if vib is not None and vib > 1.1:
            return "Critical"
        elif egt is not None and egt > 910:
            return "Degrading"
        elif n2 is not None and n2 < 520:
            return "Degrading"
        else:
            return "Healthy"

    return df.apply(classify_row, axis=1)


# Example usage
if __name__ == "__main__":
    test_df = pd.DataFrame({
        "sensor_2": [523, 519, 600],
        "sensor_3": [905, 915, 890],
        "sensor_4": [1.2, 0.9, 1.0]
    })
    test_df["Rule_Predicted"] = rule_based_health_label(test_df)
    print(test_df)
