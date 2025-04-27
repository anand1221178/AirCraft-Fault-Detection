# run_experiments.py

import pandas as pd
import numpy as np
import os
import time
import sys
import random

# --- Add project subdirectories to Python's search path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes run_experiments.py is in the 'Codes' directory alongside subdirs
sys.path.append(os.path.join(script_dir, 'DBN'))
sys.path.append(os.path.join(script_dir, 'PreProcessing'))
sys.path.append(os.path.join(script_dir, 'Utils'))
sys.path.append(os.path.join(script_dir, 'Data_Gen')) 

# --- Imports ---
try:
    # Simulation
    from sim_data import simulate_engine_data # Assume main simulation function
    from config import PARAMS as discretization_params, SIMULATION_DURATION_MINUTES, DATA_FREQUENCY_HZ
    # Discretization
    from discretize_data import discretize_data 
    # MRF
    from mrf_model import apply_simple_mrf_smoother
    # DBN Models
    from dbn_model import define_dbn_structure as define_full_dbn_structure
    from dbn_model import define_initial_cpts as define_full_initial_cpts
    # Incorporate Vanilla DBN functions here or import from vanilla_dbn_model.py
    from vanilla_dbn import define_vanilla_dbn_structure, define_vanilla_initial_cpts 
    # Inference
    from dbn_inference import prepare_evidence_sequence, run_dbn_inference, format_results_to_dataframe, OBS_STATE_MAP, OBSERVATION_VARS 
    #from dbn_inference import FULL_DBN_HIDDEN_VARS, FULL_DBN_STATE_NAMES # Use maps defined there
    #from dbn_inference import VANILLA_DBN_HIDDEN_VARS, VANILLA_DBN_STATE_NAMES
    # Prediction Logic
    from utils import map_probabilities_to_predictions 
    # Rule-Based Baseline
    from baselines import predict_rule_based # Assuming moved to baselines.py
    
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Check file locations, names, and ensure __init__.py files exist.")
    exit()
except FileNotFoundError as e:
     print(f"Error: A required file (like config.py or model file) not found: {e}")
     exit()

FULL_DBN_HIDDEN_VARS = [
    'Engine_Core_Health', 
    'Lubrication_System_Health',
    'EGT_Sensor_Health',        
    'Vibration_Sensor_Health'   
]
FULL_DBN_STATE_NAMES = {
    'Engine_Core_Health': ['OK', 'Warn', 'Fail'],
    'Lubrication_System_Health': ['OK', 'Fail'],
    'EGT_Sensor_Health': ['OK', 'Degraded', 'Failed'],     
    'Vibration_Sensor_Health': ['OK', 'Degraded', 'Failed'] 
}

VANILLA_DBN_HIDDEN_VARS = [
    'Engine_Core_Health', 
    'Lubrication_System_Health'
]
VANILLA_DBN_STATE_NAMES = {
    'Engine_Core_Health': ['OK', 'Warn', 'Fail'],
    'Lubrication_System_Health': ['OK', 'Fail']
}
# Make sure OBS_STATE_MAP and OBSERVATION_VARS are still imported or defined here if needed globally
# (It seems they are imported from dbn_inference correctly, which is fine)


# --- Experiment Configuration ---
N_RUNS = 1 # Number of simulation runs per scenario (adjust as needed)
SCENARIOS = ['Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail']
# Output file
RESULTS_DIR = os.path.join(script_dir, 'Data', 'experiment_results')
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiment_results_raw.csv')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- Model/Pipeline Parameters ---
mrf_parameters = {'iterations': 5, 'strength': 0.5}
prediction_parameters = { # Thresholds used by map_probabilities_to_predictions
    'core_warn_thresh': 0.5, 'core_fail_thresh': 0.7, 
    'lub_fail_thresh': 0.6, 
    'egt_sh_fail_thresh': 0.7, 'vib_sh_fail_thresh': 0.7
}
# Create params dict for vanilla DBN predictions (excluding sensor health thresholds)
vanilla_prediction_params = {k:v for k,v in prediction_parameters.items() if 'sh_fail' not in k}

rule_based_parameters = {'oil_low_thresh_steps': 5, 'vib_high_thresh_steps': 5}


# --- Main Experiment Loop ---
if __name__ == "__main__":
    
    all_results = [] # List to store results DataFrames from each run

    # --- Initialize DBN Models (Once) ---
    print("--- Initializing DBN Models ---")
    try:
        dbn_full = define_full_dbn_structure()
        cpt_list_full = define_full_initial_cpts()
        for cpt in cpt_list_full: dbn_full.add_cpds(cpt)
        dbn_full.check_model()
        print("Full DBN initialized.")
        
        dbn_vanilla = define_vanilla_dbn_structure()
        cpt_list_vanilla = define_vanilla_initial_cpts()
        for cpt in cpt_list_vanilla: dbn_vanilla.add_cpds(cpt)
        dbn_vanilla.check_model()
        print("Vanilla DBN initialized.")
    except Exception as e:
        print(f"Error initializing DBN models: {e}")
        exit()
        
    # --- Start Experiment Runs ---
    experiment_start_time = time.time()
    print(f"\n--- Starting Experiment ({N_RUNS} runs per scenario) ---")

    for scenario in SCENARIOS:
        print(f"\n--- Processing Scenario: {scenario} ---")
        for run_id in range(N_RUNS):
            run_start_time = time.time()
            print(f"  --- Starting Run ID: {run_id} ---")
            
            # 1. Simulate Data
            print("    Simulating data...")
            # Use run_id as seed for reproducibility if sim_data supports it
            df_raw = simulate_engine_data(
                duration_minutes=SIMULATION_DURATION_MINUTES,
                frequency_hz=DATA_FREQUENCY_HZ,
                scenario=scenario,
                # Add seed=run_id if your simulation function accepts it
            )
            
            # --- Run Full Pipeline (MRF + Integrated DBN) ---
            print("    Running Full Pipeline (MRF + Integrated DBN)...")
            predictions_full = pd.Series(dtype=str)
            try:
                # a) MRF
                vib1_smoothed, vib2_smoothed = apply_simple_mrf_smoother(df_raw['Vib1_IPS'], df_raw['Vib2_IPS'], **mrf_parameters)
                df_raw_mrf = df_raw.copy()
                df_raw_mrf['Vib1_IPS_Smoothed'] = vib1_smoothed
                df_raw_mrf['Vib2_IPS_Smoothed'] = vib2_smoothed
                # b) Discretize (Smoothed Vib)
                df_discrete_smoothed = discretize_data(df_raw_mrf, discretization_params)
                # c) DBN Inference (Full Model)
                evidence_full = prepare_evidence_sequence(df_discrete_smoothed, OBSERVATION_VARS, OBS_STATE_MAP)
                if evidence_full:
                    inference_results_full = run_dbn_inference(dbn_full, evidence_full, FULL_DBN_HIDDEN_VARS)
                    results_full_dbn = format_results_to_dataframe(inference_results_full, FULL_DBN_HIDDEN_VARS, FULL_DBN_STATE_NAMES)
                    # d) Prediction Mapping
                    predictions_full = map_probabilities_to_predictions(results_full_dbn, **prediction_parameters)
                else: predictions_full = pd.Series('Error_NoEvidence', index=df_raw.index)
            except Exception as e:
                 print(f"      ERROR in Full Pipeline: {e}")
                 predictions_full = pd.Series('Error_PipelineFail', index=df_raw.index)


            # --- Run Vanilla DBN Pipeline (No MRF, No Sensor Health) ---
            print("    Running Vanilla DBN Pipeline...")
            predictions_vanilla = pd.Series(dtype=str)
            try:
                # a) Discretize (Raw Vib)
                df_discrete_raw = discretize_data(df_raw, discretization_params) # Use original raw data
                # b) DBN Inference (Vanilla Model)
                evidence_vanilla = prepare_evidence_sequence(df_discrete_raw, OBSERVATION_VARS, OBS_STATE_MAP)
                if evidence_vanilla:
                    inference_results_vanilla = run_dbn_inference(dbn_vanilla, evidence_vanilla, VANILLA_DBN_HIDDEN_VARS)
                    results_vanilla_dbn = format_results_to_dataframe(inference_results_vanilla, VANILLA_DBN_HIDDEN_VARS, VANILLA_DBN_STATE_NAMES)
                    # c) Prediction Mapping (using simplified params/logic)
                    predictions_vanilla = map_probabilities_to_predictions(results_vanilla_dbn, **vanilla_prediction_params)
                else: predictions_vanilla = pd.Series('Error_NoEvidence', index=df_raw.index)
            except Exception as e:
                 print(f"      ERROR in Vanilla Pipeline: {e}")
                 predictions_vanilla = pd.Series('Error_PipelineFail', index=df_raw.index)


            # --- Run Rule-Based Model (Uses Smoothed Discrete Data) ---
            print("    Running Rule-Based Model...")
            predictions_rule = pd.Series(dtype=str)
            try:
                # Need smoothed discrete data from Full pipeline run (step 2a)
                if 'df_discrete_smoothed' in locals():
                     predictions_rule = predict_rule_based(df_discrete_smoothed, **rule_based_parameters)
                else:
                     # Fallback: discretize again if needed (less efficient)
                     print("      Re-discretizing for rule-based (fallback)...")
                     if 'df_raw_mrf' not in locals(): # Need MRF output first
                          vib1_s, vib2_s = apply_simple_mrf_smoother(df_raw['Vib1_IPS'], df_raw['Vib2_IPS'], **mrf_parameters)
                          df_raw_mrf = df_raw.copy(); df_raw_mrf['Vib1_IPS_Smoothed']=vib1_s; df_raw_mrf['Vib2_IPS_Smoothed']=vib2_s
                     df_discrete_smoothed_rules = discretize_data(df_raw_mrf, discretization_params)
                     predictions_rule = predict_rule_based(df_discrete_smoothed_rules, **rule_based_params)

            except Exception as e:
                 print(f"      ERROR in Rule-Based Model: {e}")
                 predictions_rule = pd.Series('Error_PipelineFail', index=df_raw.index)


            # --- Combine results for this run ---
            print("    Combining run results...")
            run_results_df = df_raw[['Timestamp', 'Scenario', 'Engine_Fault_State', 'EGT_Sensor_Health', 'Vibration_Sensor_Health']].copy()
            run_results_df['RunID'] = run_id
            run_results_df['TimeStep'] = range(len(run_results_df))
            
            # Ensure indices align before assigning predictions
            run_results_df.reset_index(drop=True, inplace=True)
            predictions_full.index = run_results_df.index[:len(predictions_full)]
            predictions_vanilla.index = run_results_df.index[:len(predictions_vanilla)]
            predictions_rule.index = run_results_df.index[:len(predictions_rule)]
            
            run_results_df['Prediction_FullDBN'] = predictions_full
            run_results_df['Prediction_VanillaDBN'] = predictions_vanilla
            run_results_df['Prediction_RuleBased'] = predictions_rule
            
            all_results.append(run_results_df)
            
            run_end_time = time.time()
            print(f"  --- Run ID: {run_id} finished in {run_end_time - run_start_time:.2f} seconds ---")


    # --- Consolidate and Save All Results ---
    print("\n--- Consolidating all experiment results ---")
    if not all_results:
        print("ERROR: No results were generated.")
    else:
        final_experiment_df = pd.concat(all_results, ignore_index=True)
        print(f"Final results DataFrame shape: {final_experiment_df.shape}")
        print("Saving final results...")
        final_experiment_df.to_csv(RESULTS_FILE, index=False)
        print(f"Experiment results saved to {RESULTS_FILE}")

    experiment_end_time = time.time()
    print(f"\n--- Experiment Finished in {experiment_end_time - experiment_start_time:.2f} seconds ---")