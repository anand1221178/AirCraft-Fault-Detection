# main_script.py

import pandas as pd
import numpy as np
import os
import time 
import sys

script_dir = os.path.dirname(os.path.abspath(__file__)) 
# Add the subdirectories relative to the script's location
sys.path.append(os.path.join(script_dir, 'DBN'))
sys.path.append(os.path.join(script_dir, 'PreProcessing'))
sys.path.append(os.path.join(script_dir, 'Utils'))
sys.path.append(os.path.join(script_dir, 'Data_Gen')) 
# --- Configuration ---

# Now your regular imports should work (without the relative dots)
try:
    from dbn_model import define_dbn_structure, define_initial_cpts 
    from dbn_inference import prepare_evidence_sequence, run_dbn_inference, format_results_to_dataframe, OBS_STATE_MAP, HIDDEN_VARS_TO_QUERY, state_names, OBSERVATION_VARS
    from mrf_model import apply_simple_mrf_smoother
    from discretize_data import discretize_data # Assuming it's in Data_Gen
    from utils import map_probabilities_to_predictions 
    from config import PARAMS as discretization_params # Import PARAMS directly now
except ImportError as e:
    print(f"Error importing modules after path modification: {e}")
    print("Check filenames and function/variable names in subdirectories.")
    print("Make sure __init__.py files exist if needed by specific imports (though not for sys.path method itself).")
    exit()

DATA_DIR = 'Data' 
RAW_DATA_FILE = os.path.join(DATA_DIR, 'raw_data/sim_data_raw.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'pipeline_output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Placeholder for Discretization Function ---
# You need to implement this based on your discretization logic
def discretize_data(df_raw, params):
    """
    Placeholder for discretization. Takes raw data df, returns discrete data df.
    IMPORTANT: Use smoothed vibration data if available.
    Handles NaNs appropriately for discrete mapping.
    """
    print("Discretizing data...")
    df_discrete = df_raw.copy() # Start with a copy
    
    # Example discretization (replace with your actual logic)
    sensors_to_discretize = {
        'EGT_C': params['EGT'],
        'N2_PctRPM': params['N2'],
        'OilPressure_PSI': params['OilPressure'],
        'Vib1_IPS': params['Vibration'], # Use original params for thresholds
        'Vib2_IPS': params['Vibration']
    }
    
    # Use smoothed columns if they exist, otherwise use raw
    vib1_col = 'Vib1_IPS_Smoothed' if 'Vib1_IPS_Smoothed' in df_discrete.columns else 'Vib1_IPS'
    vib2_col = 'Vib2_IPS_Smoothed' if 'Vib2_IPS_Smoothed' in df_discrete.columns else 'Vib2_IPS'

    for col, p in sensors_to_discretize.items():
        discrete_col_name = col + '_Discrete'
        raw_col_to_use = col # Default
        if col == 'Vib1_IPS': raw_col_to_use = vib1_col
        if col == 'Vib2_IPS': raw_col_to_use = vib2_col
            
        low_thresh = p['low_thresh']
        high_thresh = p['high_thresh']
        
        conditions = [
            df_discrete[raw_col_to_use] < low_thresh,
            (df_discrete[raw_col_to_use] >= low_thresh) & (df_discrete[raw_col_to_use] <= high_thresh),
            df_discrete[raw_col_to_use] > high_thresh
        ]
        choices = ['Low', 'Medium', 'High']
        df_discrete[discrete_col_name] = np.select(conditions, choices, default='Missing') # Or handle NaN separately

        # Ensure NaNs in raw map to 'Missing' or NaN
        df_discrete.loc[df_discrete[raw_col_to_use].isna(), discrete_col_name] = np.nan # Or 'Missing'

    print("Discretization complete.")
    return df_discrete[['Timestamp', 'Scenario', 'Engine_Fault_State', 'EGT_Sensor_Health', 'Vibration_Sensor_Health'] + [c + '_Discrete' for c in sensors_to_discretize.keys()]]


# --- Main Pipeline Function ---
def run_full_pipeline(df_raw_slice, dbn_model, mrf_params, discretization_params, prediction_params):
    """Runs the full MRF -> Discretize -> DBN -> Predict pipeline."""
    
    start_time = time.time()
    print(f"\n--- Running Pipeline on Slice (Length: {len(df_raw_slice)}) ---")

    # 1. Apply MRF Smoothing
    vib1_smoothed, vib2_smoothed = apply_simple_mrf_smoother(
        df_raw_slice['Vib1_IPS'], 
        df_raw_slice['Vib2_IPS'],
        iterations=mrf_params.get('iterations', 5),
        strength=mrf_params.get('strength', 0.5)
    )
    # Add smoothed data back to the slice (or a copy)
    df_slice_processed = df_raw_slice.copy()
    df_slice_processed['Vib1_IPS_Smoothed'] = vib1_smoothed
    df_slice_processed['Vib2_IPS_Smoothed'] = vib2_smoothed
    
    # 2. Discretize Data (using smoothed Vib)
    df_discrete_slice = discretize_data(df_slice_processed, discretization_params)

    # 3. Prepare DBN Evidence
    evidence_sequence = prepare_evidence_sequence(df_discrete_slice, OBSERVATION_VARS, OBS_STATE_MAP)
    if not evidence_sequence:
        print("Error: No evidence generated from slice. Aborting pipeline.")
        return None, None

    # 4. Run DBN Inference
    inference_results = run_dbn_inference(dbn_model, evidence_sequence, HIDDEN_VARS_TO_QUERY)

    # 5. Format Inference Results
    results_df = format_results_to_dataframe(inference_results, HIDDEN_VARS_TO_QUERY, state_names)

    # 6. Map Probabilities to Predictions
    final_predictions = map_probabilities_to_predictions(results_df, **prediction_params)

    end_time = time.time()
    print(f"--- Pipeline Finished in {end_time - start_time:.2f} seconds ---")
    
    # Add predictions to the results DataFrame
    results_df['Final_Prediction'] = final_predictions
    
    # Include Ground Truth for easy comparison later
    gt_cols_to_add = ['Engine_Fault_State', 'EGT_Sensor_Health', 'Vibration_Sensor_Health']
    for col in gt_cols_to_add:
         if col in df_discrete_slice.columns:
              results_df[f'GT_{col}'] = df_discrete_slice[col].values[:len(results_df)] # Ensure length match

    return results_df


# --- Example Execution ---
if __name__ == "__main__":
    
    # --- Load DBN Model (Do this once) ---
    print("--- Initializing DBN Model ---")
    try:
        dbn = define_dbn_structure()
        cpt_list = define_initial_cpts()
        for cpt in cpt_list: dbn.add_cpds(cpt)
        dbn.check_model()
        print("DBN model initialized and checked.")
    except Exception as e:
        print(f"Error initializing DBN: {e}")
        exit()

    # --- Load Raw Data ---
    print(f"\n--- Loading Raw Data: {RAW_DATA_FILE} ---")
    try:
        df_raw_full = pd.read_csv(RAW_DATA_FILE, parse_dates=['Timestamp'])
        print(f"Loaded data shape: {df_raw_full.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {RAW_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"Error loading raw data: {e}")
        exit()

    # --- Define Parameters ---
    # Import params from config or define here
    # Example: PARAMS from config.py would be used by discretize_data
    try:
        from config import PARAMS as discretization_params # Assuming config.py is accessible
    except ImportError:
        print("Warning: config.py not found. Using dummy discretization params.")
        discretization_params = {} # Add dummy thresholds if needed
        
    mrf_parameters = {'iterations': 5, 'strength': 0.5}
    prediction_parameters = { # Thresholds for converting probs to predictions
        'core_warn_thresh': 0.5, 
        'core_fail_thresh': 0.7, 
        'lub_fail_thresh': 0.6, 
        'egt_sh_fail_thresh': 0.7, 
        'vib_sh_fail_thresh': 0.7
    }

    # --- Test Pipeline on a Short Slice ---
    # Select a slice for testing (e.g., first 100 steps of a specific scenario)
    test_scenario = 'EGTSensorFail'
    test_raw_slice = df_raw_full[df_raw_full['Scenario'] == test_scenario].head(100).copy()
    
    if test_raw_slice.empty:
         print(f"No data for scenario '{test_scenario}' to test pipeline.")
    else:
        print(f"\n--- Testing Pipeline on '{test_scenario}' Slice ---")
        pipeline_results_df = run_full_pipeline(
            test_raw_slice, 
            dbn, 
            mrf_parameters,
            discretization_params, # Pass the PARAMS dict here
            prediction_parameters
        )

        if pipeline_results_df is not None:
            print("\n--- Pipeline Test Output (Sample) ---")
            print(pipeline_results_df.head())
            print(pipeline_results_df.tail())
            
            # Save the output of this test run
            output_filename = os.path.join(OUTPUT_DIR, f'pipeline_test_{test_scenario}.csv')
            pipeline_results_df.to_csv(output_filename, index=False)
            print(f"Saved pipeline test output to {output_filename}")

    print("\n--- Main Script Finished ---")