import pandas as pd
import numpy as np

def map_probabilities_to_predictions(results_df, 
                                     core_warn_thresh=0.5, 
                                     core_fail_thresh=0.7, # Might not be reached often
                                     lub_fail_thresh=0.6, 
                                     egt_sh_fail_thresh=0.7, 
                                     vib_sh_fail_thresh=0.7):
    """
    Converts DBN probability outputs into discrete fault/health predictions.
    Uses a hierarchical logic: check critical failures first, then sensor failures.

    Args:
        results_df (pd.DataFrame): DataFrame from format_results_to_dataframe, 
                                   containing P(State) columns.
        core_warn_thresh (float): Threshold for P(CoreHealth=Warn).
        core_fail_thresh (float): Threshold for P(CoreHealth=Fail).
        lub_fail_thresh (float): Threshold for P(LubHealth=Fail).
        egt_sh_fail_thresh (float): Threshold for P(EGT SensorHealth=Failed).
        vib_sh_fail_thresh (float): Threshold for P(Vib SensorHealth=Failed).

    Returns:
        pd.Series: A series with the final discrete prediction for each timestep.
    """
    # Initialize an empty series to store predictions for each time step
    predictions = pd.Series(index=results_df.index, dtype=str)
    
    # Define probability columns expected (handle potential missing columns gracefully)
    p_core_ok   = f"P(Engine_Core_Health=OK)"
    p_core_warn = f"P(Engine_Core_Health=Warn)"
    p_core_fail = f"P(Engine_Core_Health=Fail)"
    p_lub_ok    = f"P(Lubrication_System_Health=OK)"
    p_lub_fail  = f"P(Lubrication_System_Health=Fail)"
    p_egt_sh_ok = f"P(EGT_Sensor_Health=OK)"
    p_egt_sh_d  = f"P(EGT_Sensor_Health=Degraded)"
    p_egt_sh_f  = f"P(EGT_Sensor_Health=Failed)"
    p_vib_sh_ok = f"P(Vibration_Sensor_Health=OK)"
    p_vib_sh_d  = f"P(Vibration_Sensor_Health=Degraded)"
    p_vib_sh_f  = f"P(Vibration_Sensor_Health=Failed)"

    # Iterate through each row in the results DataFrame
    for index, row in results_df.iterrows():
        # Check for critical component failures first
        if p_lub_fail in row and row[p_lub_fail] > lub_fail_thresh:
            predictions.loc[index] = 'OilLeak_Predicted'
        # Check for Core Fail OR high Warn state
        elif (p_core_fail in row and row[p_core_fail] > core_fail_thresh) or \
             (p_core_warn in row and row[p_core_warn] > core_warn_thresh): 
             # Prioritize Core Warn/Fail if both LubFail and CoreWarn/Fail are high? Depends on logic.
             # Current logic: LubFail takes precedence if its threshold is met.
            predictions.loc[index] = 'BearingWear_Predicted'
        # If no major component fault, check for sensor failures
        elif p_egt_sh_f in row and row[p_egt_sh_f] > egt_sh_fail_thresh:
            predictions.loc[index] = 'EGTSensorFail_Predicted'
        elif p_vib_sh_f in row and row[p_vib_sh_f] > vib_sh_fail_thresh:
             predictions.loc[index] = 'VibSensorFail_Predicted'
        # If none of the above, assume Normal
        else:
            predictions.loc[index] = 'Normal_Predicted'
            
    return predictions

# Example usage (if run standalone for testing)
if __name__ == "__main__":
     # Create dummy data matching the output of format_results_to_dataframe
     dummy_data = {
         'TimeStep': [0, 1, 2, 3, 4],
         'P(Engine_Core_Health=OK)': [0.9, 0.8, 0.2, 0.1, 0.1],
         'P(Engine_Core_Health=Warn)': [0.08, 0.15, 0.7, 0.8, 0.1],
         'P(Engine_Core_Health=Fail)': [0.02, 0.05, 0.1, 0.1, 0.8],
         'P(Lubrication_System_Health=OK)': [0.99, 0.3, 0.9, 0.9, 0.9],
         'P(Lubrication_System_Health=Fail)': [0.01, 0.7, 0.1, 0.1, 0.1],
         'P(EGT_Sensor_Health=OK)': [0.9, 0.9, 0.9, 0.2, 0.9],
         'P(EGT_Sensor_Health=Degraded)': [0.05, 0.05, 0.05, 0.1, 0.05],
         'P(EGT_Sensor_Health=Failed)': [0.05, 0.05, 0.05, 0.7, 0.05],
         'P(Vibration_Sensor_Health=OK)': [0.9, 0.9, 0.9, 0.9, 0.2],
         'P(Vibration_Sensor_Health=Degraded)': [0.05, 0.05, 0.05, 0.05, 0.1],
         'P(Vibration_Sensor_Health=Failed)': [0.05, 0.05, 0.05, 0.05, 0.7]
     }
     dummy_results_df = pd.DataFrame(dummy_data)
     dummy_results_df.set_index('TimeStep', inplace=True)

     # Print the dummy results DataFrame
     print("Dummy Results DF:\n", dummy_results_df)
     
     # Generate predictions based on the dummy data
     final_preds = map_probabilities_to_predictions(dummy_results_df)
     print("\nFinal Predictions:\n", final_preds)
     # Expected: Normal, OilLeak, BearingWear, EGTSensorFail, VibSensorFail