# main_script.py
###----------------------------- NOTE: THIS FILE IS NOT USED !!! -------------------
import pandas as pd
import numpy as np
import os
import time 
import sys 
from pgmpy.models import DynamicBayesianNetwork # Need this for Vanilla DBN
from pgmpy.factors.discrete import TabularCPD # Need this for Vanilla DBN

# --- Add project subdirectories to Python's search path ---
script_dir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(script_dir, 'DBN'))
sys.path.append(os.path.join(script_dir, 'PreProcessing'))
sys.path.append(os.path.join(script_dir, 'Utils'))
sys.path.append(os.path.join(script_dir, 'Data_Gen')) 

# --- Imports ---
try:
    # Full DBN components
    from dbn_model import define_dbn_structure as define_full_dbn_structure
    from dbn_model import define_initial_cpts as define_full_initial_cpts
    from dbn_inference import prepare_evidence_sequence, run_dbn_inference, format_results_to_dataframe, OBS_STATE_MAP, OBSERVATION_VARS
    # Note: HIDDEN_VARS_TO_QUERY and state_names will be defined below based on model
    
    # MRF
    from mrf_model import apply_simple_mrf_smoother
    
    # Discretization (assuming function is in Data_Gen/discretize_data.py)
    from discretize_data import discretize_data 
    
    # Prediction Logic
    from utils import map_probabilities_to_predictions 
    
    # Config
    from config import PARAMS as discretization_params 
except ImportError as e:
    print(f"Error importing modules after path modification: {e}")
    print("Check filenames/functions/variables in subdirectories.")
    exit()

# --- Configuration ---
DATA_DIR = 'Data' 
RAW_DATA_FILE = os.path.join(DATA_DIR, 'raw_data/sim_data_raw.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'pipeline_output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- State Names Maps (Define globally for different models) ---
FULL_DBN_HIDDEN_VARS = [
    'Engine_Core_Health', 'Lubrication_System_Health',
    'EGT_Sensor_Health', 'Vibration_Sensor_Health'   
]
FULL_DBN_STATE_NAMES = {
    'Engine_Core_Health': ['OK', 'Warn', 'Fail'],
    'Lubrication_System_Health': ['OK', 'Fail'],
    'EGT_Sensor_Health': ['OK', 'Degraded', 'Failed'],     
    'Vibration_Sensor_Health': ['OK', 'Degraded', 'Failed'] 
}

VANILLA_DBN_HIDDEN_VARS = [
    'Engine_Core_Health', 'Lubrication_System_Health'
]
VANILLA_DBN_STATE_NAMES = {
    'Engine_Core_Health': ['OK', 'Warn', 'Fail'],
    'Lubrication_System_Health': ['OK', 'Fail']
}


# === VANILLA DBN MODEL DEFINITION (Incorporated from vanilla_dbn_model.py) ===
def define_vanilla_dbn_structure():
    """Defines a simpler DBN structure WITHOUT sensor health nodes."""
    dbn_structure = [
        (('Engine_Core_Health', 0), ('EGT_C_Discrete', 0)), (('Engine_Core_Health', 0), ('N2_PctRPM_Discrete', 0)),
        (('Engine_Core_Health', 0), ('Vib1_IPS_Discrete', 0)), (('Engine_Core_Health', 0), ('Vib2_IPS_Discrete', 0)),
        (('Lubrication_System_Health', 0), ('OilPressure_PSI_Discrete', 0)),
        (('Engine_Core_Health', 0), ('Engine_Core_Health', 1)), (('Lubrication_System_Health', 0), ('Lubrication_System_Health', 1)),
        (('Engine_Core_Health', 1), ('EGT_C_Discrete', 1)), (('Engine_Core_Health', 1), ('N2_PctRPM_Discrete', 1)),
        (('Engine_Core_Health', 1), ('Vib1_IPS_Discrete', 1)), (('Engine_Core_Health', 1), ('Vib2_IPS_Discrete', 1)),
        (('Lubrication_System_Health', 1), ('OilPressure_PSI_Discrete', 1)),
    ]
    dbn = DynamicBayesianNetwork(dbn_structure)
    return dbn

def define_vanilla_initial_cpts():
    """Defines CPTs for the Vanilla DBN."""
    cpt_list = []
    # Initial States
    cpd_core_health_0 = TabularCPD(variable=('Engine_Core_Health', 0), variable_card=3, values=[[0.95], [0.04], [0.01]])
    cpt_list.append(cpd_core_health_0)
    cpd_lub_health_0 = TabularCPD(variable=('Lubrication_System_Health', 0), variable_card=2, values=[[0.98], [0.02]])
    cpt_list.append(cpd_lub_health_0)
    # Observations t=0
    egt_obs_values = [[0.10, 0.10, 0.05], [0.80, 0.50, 0.25], [0.10, 0.40, 0.70]] 
    cpd_egt_obs = TabularCPD(variable=('EGT_C_Discrete', 0), variable_card=3, values=egt_obs_values, evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_egt_obs)
    n2_obs_values = [[0.10, 0.10, 0.15], [0.80, 0.80, 0.75], [0.10, 0.10, 0.10]] 
    cpd_n2_obs = TabularCPD(variable=('N2_PctRPM_Discrete', 0), variable_card=3, values=n2_obs_values, evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_n2_obs)
    oilp_obs_values = [[0.10, 0.80], [0.80, 0.15], [0.10, 0.05]] 
    cpd_oilp_obs = TabularCPD(variable=('OilPressure_PSI_Discrete', 0), variable_card=3, values=oilp_obs_values, evidence=[('Lubrication_System_Health', 0)], evidence_card=[2])
    cpt_list.append(cpd_oilp_obs)
    vib_obs_values = [[0.60, 0.10, 0.05], [0.35, 0.60, 0.25], [0.05, 0.30, 0.70]] 
    cpd_vib1_obs = TabularCPD(variable=('Vib1_IPS_Discrete', 0), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_vib1_obs)
    cpd_vib2_obs = TabularCPD(variable=('Vib2_IPS_Discrete', 0), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_vib2_obs)
    # Transitions
    cpd_core_health_t1 = TabularCPD(variable=('Engine_Core_Health', 1), variable_card=3, values=[[0.95, 0.10, 0.01], [0.04, 0.85, 0.19], [0.01, 0.05, 0.80]], evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_core_health_t1)
    cpd_lub_health_t1 = TabularCPD(variable=('Lubrication_System_Health', 1), variable_card=2, values=[[0.98, 0.00], [0.02, 1.00]], evidence=[('Lubrication_System_Health', 0)], evidence_card=[2])
    cpt_list.append(cpd_lub_health_t1)
    # Observations t=1
    cpd_egt_obs_t1 = TabularCPD(variable=('EGT_C_Discrete', 1), variable_card=3, values=egt_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_egt_obs_t1)
    cpd_n2_obs_t1 = TabularCPD(variable=('N2_PctRPM_Discrete', 1), variable_card=3, values=n2_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_n2_obs_t1)
    cpd_oilp_obs_t1 = TabularCPD(variable=('OilPressure_PSI_Discrete', 1), variable_card=3, values=oilp_obs_values, evidence=[('Lubrication_System_Health', 1)], evidence_card=[2])
    cpt_list.append(cpd_oilp_obs_t1)
    cpd_vib1_obs_t1 = TabularCPD(variable=('Vib1_IPS_Discrete', 1), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_vib1_obs_t1)
    cpd_vib2_obs_t1 = TabularCPD(variable=('Vib2_IPS_Discrete', 1), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_vib2_obs_t1)
    # print(f"Defined {len(cpt_list)} CPTs for the Vanilla DBN model.") # Optional print
    return cpt_list
# === END VANILLA DBN DEFINITION ===


# === RULE-BASED MODEL DEFINITION (Incorporated from Utils/baselines.py) ===
def predict_rule_based(df_discrete_slice, 
                       oil_low_thresh_steps=5, 
                       vib_high_thresh_steps=5):
    """Applies simple threshold-based rules to predict faults."""
    predictions = pd.Series('Normal_Predicted', index=df_discrete_slice.index) 
    is_oil_low = (df_discrete_slice['OilPressure_PSI_Discrete'] == 'Low')
    oil_low_consecutive = is_oil_low.rolling(window=oil_low_thresh_steps, min_periods=oil_low_thresh_steps).sum()
    oil_leak_indices = oil_low_consecutive[oil_low_consecutive >= oil_low_thresh_steps].index
    predictions.loc[oil_leak_indices] = 'OilLeak_Predicted'
    is_vib_high = (df_discrete_slice['Vib1_IPS_Discrete'] == 'High') | (df_discrete_slice['Vib2_IPS_Discrete'] == 'High')
    vib_high_consecutive = is_vib_high.rolling(window=vib_high_thresh_steps, min_periods=vib_high_thresh_steps).sum()
    bearing_wear_indices = vib_high_consecutive[vib_high_consecutive >= vib_high_thresh_steps].index
    predictions.loc[bearing_wear_indices.difference(oil_leak_indices)] = 'BearingWear_Predicted'
    return predictions
# === END RULE-BASED DEFINITION ===


# === PIPELINE FUNCTION (Modified to handle different models) ===
def run_analysis_pipeline(df_raw_slice, 
                          full_dbn_model, 
                          vanilla_dbn_model,
                          mrf_params, 
                          discretization_params, 
                          full_dbn_prediction_params,
                          vanilla_dbn_prediction_params,
                          rule_based_params):
    """
    Runs the full analysis pipeline including all models.

    Returns:
        pd.DataFrame: Combined results with GT and predictions from all models.
                      Returns None if a critical step fails.
    """
    start_time_total = time.time()
    print(f"\n--- Running Analysis Pipeline on Slice (Length: {len(df_raw_slice)}) ---")

    # --- Step 1: MRF Smoothing (for Full DBN and Rule-Based) ---
    print("Applying MRF smoother...")
    vib1_smoothed, vib2_smoothed = apply_simple_mrf_smoother(
        df_raw_slice['Vib1_IPS'], 
        df_raw_slice['Vib2_IPS'],
        iterations=mrf_params.get('iterations', 5),
        strength=mrf_params.get('strength', 0.5)
    )
    df_slice_mrf = df_raw_slice.copy()
    df_slice_mrf['Vib1_IPS_Smoothed'] = vib1_smoothed
    df_slice_mrf['Vib2_IPS_Smoothed'] = vib2_smoothed

    # --- Step 2: Discretization ---
    # a) Using smoothed data (for Full DBN, Rule-Based)
    print("Discretizing data with smoothed vibration...")
    df_discrete_smoothed = discretize_data(df_slice_mrf, discretization_params)
    if df_discrete_smoothed is None or df_discrete_smoothed.empty: return None # Added check

    # b) Using raw data (for Vanilla DBN)
    print("Discretizing data with raw vibration...")
    df_discrete_raw = discretize_data(df_raw_slice, discretization_params) # Pass original raw slice
    if df_discrete_raw is None or df_discrete_raw.empty: return None # Added check

    # --- Step 3: Run Full DBN Pipeline ---
    print("\n--- Running Full DBN Inference ---")
    results_full_dbn = None
    try:
        evidence_full = prepare_evidence_sequence(df_discrete_smoothed, OBSERVATION_VARS, OBS_STATE_MAP)
        if evidence_full:
            inference_results_full = run_dbn_inference(full_dbn_model, evidence_full, FULL_DBN_HIDDEN_VARS)
            results_full_dbn = format_results_to_dataframe(inference_results_full, FULL_DBN_HIDDEN_VARS, FULL_DBN_STATE_NAMES)
            predictions_full = map_probabilities_to_predictions(results_full_dbn, **full_dbn_prediction_params)
            results_full_dbn['Prediction_FullDBN'] = predictions_full
        else: print("No evidence for Full DBN.")
    except Exception as e:
        print(f"Error during Full DBN pipeline: {e}")
        # Continue to other models

    # --- Step 4: Run Vanilla DBN Pipeline ---
    print("\n--- Running Vanilla DBN Inference ---")
    results_vanilla_dbn = None
    try:
        # Need to adapt prediction function if state names differ
        vanilla_pred_params = {k:v for k,v in full_dbn_prediction_params.items() if 'sh_fail' not in k}
        
        evidence_vanilla = prepare_evidence_sequence(df_discrete_raw, OBSERVATION_VARS, OBS_STATE_MAP) # Use raw discrete data
        if evidence_vanilla:
            inference_results_vanilla = run_dbn_inference(vanilla_dbn_model, evidence_vanilla, VANILLA_DBN_HIDDEN_VARS) # Use vanilla model/vars
            results_vanilla_dbn = format_results_to_dataframe(inference_results_vanilla, VANILLA_DBN_HIDDEN_VARS, VANILLA_DBN_STATE_NAMES) # Use vanilla state names
            # Adapt map_probabilities_to_predictions or create a vanilla version
            # For now, let's assume the existing one works if columns are present
            predictions_vanilla = map_probabilities_to_predictions(results_vanilla_dbn, **vanilla_pred_params) # Use adapted params
            results_vanilla_dbn['Prediction_VanillaDBN'] = predictions_vanilla
        else: print("No evidence for Vanilla DBN.")
    except Exception as e:
        print(f"Error during Vanilla DBN pipeline: {e}")
        # Continue to other models
        
    # --- Step 5: Run Rule-Based Model ---
    print("\n--- Running Rule-Based Model ---")
    predictions_rule = None
    try:
        # Use discrete data with smoothed vibration
        predictions_rule = predict_rule_based(df_discrete_smoothed, **rule_based_params) 
    except Exception as e:
        print(f"Error during Rule-Based prediction: {e}")
        # Continue
        
    # --- Step 6: Combine Results ---
    print("\n--- Combining Results ---")
    # Use the smoothed discrete data as the base for ground truth and index
    final_results_df = df_discrete_smoothed[['Timestamp','Scenario','Engine_Fault_State','EGT_Sensor_Health','Vibration_Sensor_Health']].copy()
    final_results_df.rename(columns={
        'Engine_Fault_State': 'GT_Engine_Fault_State',
        'EGT_Sensor_Health': 'GT_EGT_Sensor_Health',
        'Vibration_Sensor_Health': 'GT_Vibration_Sensor_Health'
    }, inplace=True)
    
    # Add predictions if they were generated successfully
    if results_full_dbn is not None and 'Prediction_FullDBN' in results_full_dbn.columns:
        final_results_df['Prediction_FullDBN'] = results_full_dbn['Prediction_FullDBN'].values[:len(final_results_df)]
    else:
        final_results_df['Prediction_FullDBN'] = 'Error'
        
    if results_vanilla_dbn is not None and 'Prediction_VanillaDBN' in results_vanilla_dbn.columns:
         final_results_df['Prediction_VanillaDBN'] = results_vanilla_dbn['Prediction_VanillaDBN'].values[:len(final_results_df)]
    else:
        final_results_df['Prediction_VanillaDBN'] = 'Error'

    if predictions_rule is not None:
        final_results_df['Prediction_RuleBased'] = predictions_rule.values[:len(final_results_df)]
    else:
        final_results_df['Prediction_RuleBased'] = 'Error'
        
    # Optionally add probability columns from DBNs if needed for detailed analysis later
    # ... (code to merge probability columns based on index/timestep) ...

    end_time_total = time.time()
    print(f"--- Analysis Pipeline Finished in {end_time_total - start_time_total:.2f} seconds ---")
    
    return final_results_df


# === MAIN EXECUTION BLOCK ===
if __name__ == "__main__":
    
    # --- Initialize Models (Once) ---
    print("--- Initializing Full DBN Model ---")
    try:
        dbn_full = define_full_dbn_structure()
        cpt_list_full = define_full_initial_cpts()
        for cpt in cpt_list_full: dbn_full.add_cpds(cpt)
        dbn_full.check_model()
        print("Full DBN model initialized and checked.")
    except Exception as e:
        print(f"Error initializing Full DBN: {e}")
        exit()
        
    print("\n--- Initializing Vanilla DBN Model ---")
    try:
        dbn_vanilla = define_vanilla_dbn_structure()
        cpt_list_vanilla = define_vanilla_initial_cpts()
        for cpt in cpt_list_vanilla: dbn_vanilla.add_cpds(cpt)
        dbn_vanilla.check_model()
        print("Vanilla DBN model initialized and checked.")
    except Exception as e:
        print(f"Error initializing Vanilla DBN: {e}")
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
    mrf_parameters = {'iterations': 5, 'strength': 0.5}
    # Using same prediction thresholds for both DBNs for now
    # Note: Vanilla DBN won't use sensor health thresholds
    prediction_parameters = { 
        'core_warn_thresh': 0.5, 'core_fail_thresh': 0.7, 
        'lub_fail_thresh': 0.6, 
        'egt_sh_fail_thresh': 0.7, 'vib_sh_fail_thresh': 0.7
    }
    rule_based_parameters = {'oil_low_thresh_steps': 5, 'vib_high_thresh_steps': 5}

    # --- Run Pipeline on a Test Slice ---
    test_scenario = 'EGTSensorFail' # Choose scenario to test
    N_STEPS = 100 # Number of steps for the test slice
    
    # Select slice (e.g., first N steps)
    test_raw_slice = df_raw_full[df_raw_full['Scenario'] == test_scenario].head(N_STEPS).copy()
    
    if test_raw_slice.empty:
         print(f"No data for scenario '{test_scenario}' to test pipeline.")
    else:
        print(f"\n--- Testing Pipeline on '{test_scenario}' Slice ({N_STEPS} steps) ---")
        
        # Call the main analysis pipeline function
        pipeline_results_df = run_analysis_pipeline(
            df_raw_slice=test_raw_slice, 
            full_dbn_model=dbn_full, 
            vanilla_dbn_model=dbn_vanilla,
            mrf_params=mrf_parameters,
            discretization_params=discretization_params, 
            full_dbn_prediction_params=prediction_parameters, # Use full params for full model
            vanilla_dbn_prediction_params=prediction_parameters, # Pass full params, function should ignore extras
            rule_based_params=rule_based_parameters
        )

        if pipeline_results_df is not None:
            print("\n--- Pipeline Test Output (Sample) ---")
            print(pipeline_results_df.head())
            print(pipeline_results_df.tail())
            
            # Save the output of this test run
            output_filename = os.path.join(OUTPUT_DIR, f'pipeline_test_{test_scenario}_all_models.csv')
            pipeline_results_df.to_csv(output_filename, index=False)
            print(f"Saved pipeline test output to {output_filename}")

    print("\n--- Main Script Finished ---")