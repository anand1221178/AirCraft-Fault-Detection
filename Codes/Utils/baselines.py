# Utils/baselines.py (or wherever you keep helper functions)

import pandas as pd
import numpy as np

def predict_rule_based(df_discrete_slice, 
                       oil_low_thresh_steps=5, 
                       vib_high_thresh_steps=5):
    """
    Applies simple threshold-based rules to predict faults.

    Args:
        df_discrete_slice (pd.DataFrame): DataFrame containing the 
                                          discretized sensor columns.
        oil_low_thresh_steps (int): Consecutive steps OilPressure must be 'Low'.
        vib_high_thresh_steps (int): Consecutive steps Vib1 OR Vib2 must be 'High'.

    Returns:
        pd.Series: Series of predictions ('Normal_Predicted', 'OilLeak_Predicted', 
                   'BearingWear_Predicted').
    """
    predictions = pd.Series('Normal_Predicted', index=df_discrete_slice.index) # Default to Normal

    # --- Oil Leak Rule ---
    # Check for consecutive 'Low' oil pressure readings
    is_oil_low = (df_discrete_slice['OilPressure_PSI_Discrete'] == 'Low')
    # Use rolling window to count consecutive lows
    # Pad with False to handle edges correctly in rolling sum
    oil_low_consecutive = is_oil_low.rolling(window=oil_low_thresh_steps, min_periods=oil_low_thresh_steps).sum()
    oil_leak_indices = oil_low_consecutive[oil_low_consecutive >= oil_low_thresh_steps].index
    predictions.loc[oil_leak_indices] = 'OilLeak_Predicted'

    # --- Bearing Wear Rule ---
    # Check for consecutive 'High' vibration readings on EITHER sensor
    is_vib_high = (df_discrete_slice['Vib1_IPS_Discrete'] == 'High') | \
                  (df_discrete_slice['Vib2_IPS_Discrete'] == 'High')
    vib_high_consecutive = is_vib_high.rolling(window=vib_high_thresh_steps, min_periods=vib_high_thresh_steps).sum()
    bearing_wear_indices = vib_high_consecutive[vib_high_consecutive >= vib_high_thresh_steps].index
    
    # Apply bearing wear prediction ONLY if not already predicted as OilLeak (OilLeak takes precedence)
    predictions.loc[bearing_wear_indices.difference(oil_leak_indices)] = 'BearingWear_Predicted'
    
    return predictions

# Example Usage - for testing
if __name__ == "__main__":
     dummy_discrete = {
         'OilPressure_PSI_Discrete': ['Medium', 'Medium', 'Low', 'Low', 'Low', 'Low', 'Low', 'Medium', 'Medium', 'Medium'],
         'Vib1_IPS_Discrete': ['Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High', 'High'],
         'Vib2_IPS_Discrete': ['Medium', 'Medium', 'Medium', 'Medium', 'Medium', 'High', 'High', 'High', 'High', 'High'],
     }
     dummy_df = pd.DataFrame(dummy_discrete)
     
     rule_preds = predict_rule_based(dummy_df, oil_low_thresh_steps=3, vib_high_thresh_steps=3)
     print("Dummy Data:\n", dummy_df)
     print("\nRule-Based Predictions:\n", rule_preds)