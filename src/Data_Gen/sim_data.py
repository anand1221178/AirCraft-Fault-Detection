# File: Data_Gen/sim_data.py

import pandas as pd
import numpy as np
import os 

# Direct imports, assuming Data_Gen is on sys.path
from cmaps_data_loader import load_cmaps_data, add_rul_column, add_discrete_health_label
from discretize_data import discretize_data
# Ensure config.py is also in Data_Gen or another directory added to sys.path
# If config.py is in Data_Gen:
from config import DATASET_ID, BINNING_STRATEGY, N_BINS, MANUAL_BINS, OBSERVATION_NODES, SELECTED_SENSORS
# If config.py is in src/ (and src/ is not directly on sys.path but ./ is, when running from src/):
# You might need to adjust how config is found or ensure Data_Gen can see it.
# Given your sys.path.append in run_experiment.py for submodules, and if config.py is in Data_Gen,
# 'from config import ...' should work when run_experiment.py imports sim_data.

# Function definitions (prepare_cmaps_data, generate_drifted_engine) as before...
# ... Make sure the generate_drifted_engine function is the one I provided previously
#     that includes the base_dir fallback and uses os.path.join etc.
#     (The one that fixed the FileNotFoundError logic)

def prepare_cmaps_data(base_dir="Data/C-MAPSS", drift_scenario=None):
    # ... (implementation using load_cmaps_data, add_rul_column, etc.)
    # Step 1: Load data
    train_df, test_df, test_rul = load_cmaps_data(base_dir, dataset_id=DATASET_ID)

    # Step 2: Add RUL and health labels
    train_df = add_rul_column(train_df)
    train_df = add_discrete_health_label(train_df)

    for raw_sensor_name in SELECTED_SENSORS: 
        if raw_sensor_name in train_df.columns:
            print(f"Sensor: {raw_sensor_name}, Min: {train_df[raw_sensor_name].min():.2f}, Max: {train_df[raw_sensor_name].max():.2f}, StdDev: {train_df[raw_sensor_name].std():.2f}, NumUnique: {train_df[raw_sensor_name].nunique()}")
        else:
            print(f"Warning: Raw sensor {raw_sensor_name} not in train_df.")

    # Step 3: Discretize sensors
    train_df = discretize_data(train_df, config=MANUAL_BINS, strategy=BINNING_STRATEGY, n_bins=N_BINS)

    # Step 4: Apply synthetic drift if requested
    if drift_scenario == "EGT_Drift":
        print("[INFO] Applying synthetic EGT drift to training data...")
        drift_sensor = "sensor_11_disc"
        if drift_sensor in train_df.columns: 
            for unit in train_df["unit"].unique():
                unit_indices = train_df[train_df["unit"] == unit].index
                if not unit_indices.empty:
                    n_points = len(unit_indices)
                    drift = (np.arange(n_points) / (n_points -1 if n_points > 1 else 1) )
                    
                    original_vals = train_df.loc[unit_indices, drift_sensor].values
                    drifted_vals = original_vals + (drift * 1).astype(int)
                    max_bin_value = N_BINS - 1 
                    train_df.loc[unit_indices, drift_sensor] = np.clip(drifted_vals, 0, max_bin_value)
        else:
            print(f"Warning: Drift sensor {drift_sensor} not found in train_df. Skipping EGT_Drift for prepare_cmaps_data.")
    return train_df, test_df, test_rul

def generate_drifted_engine(scenario="EGT_Drift", n_steps=50, base_unit_id=1, 
                            dataset_id_for_base="FD001", base_dir=None): # base_dir will be passed from run_experiment
    
    print(f"[INFO] Generating SYNTHETIC drifted engine data for scenario: {scenario}, n_steps: {n_steps}")

    # Create a synthetic DataFrame
    cycles = np.arange(1, n_steps + 1)
    df_synthetic = pd.DataFrame({'unit': base_unit_id, 'cycle': cycles})

    # Set all OBSERVATION_NODES (discretized sensor names from config) to a "healthy" bin (e.g., 0 or 1)
    # Ensure OBSERVATION_NODES is imported from config
    healthy_bin_value = 0 # Or 1, depending on what's typical for healthy for most sensors
    for sensor_disc_name in OBSERVATION_NODES:
        df_synthetic[sensor_disc_name] = healthy_bin_value
    
    # Apply the specified drift scenario
    if scenario == "EGT_Drift":
        drift_sensor_disc = "sensor_11_disc" # Must be in OBSERVATION_NODES
        if drift_sensor_disc in df_synthetic.columns:
            max_bin_value = N_BINS - 1
            if n_steps > 1:
                drift_values_pattern = np.linspace(0, max_bin_value, n_steps).astype(int)
            else:
                drift_values_pattern = np.array([0]).astype(int) 
            df_synthetic[drift_sensor_disc] = np.clip(drift_values_pattern, 0, max_bin_value)
            print(f"  Applied SYNTHETIC AGGRESSIVE drift to '{drift_sensor_disc}' from bin 0 towards {max_bin_value}.")
            # print(f"  Drifted {drift_sensor_disc} (first 10): {df_synthetic[drift_sensor_disc].head(10).values}")
        else:
            print(f"  Warning: Drift sensor '{drift_sensor_disc}' not in OBSERVATION_NODES. Cannot apply EGT_Drift.")
            
    elif scenario == "Vibration_Spike":
        # ... (implement similar logic for a synthetic spike if needed) ...
        pass
    else:
        print(f"  Warning: Unknown drift scenario '{scenario}'. No drift applied to synthetic data.")

    # The RUL and HealthState will be added in run_experiment.py to force "Healthy"
    return df_synthetic



# def generate_drifted_engine(scenario="EGT_Drift", n_steps=None, base_unit_id=1, 
#                             dataset_id_for_base="FD001", base_dir=None):
#     print(f"[INFO] Generating drifted engine data for scenario: {scenario}, base unit: {base_unit_id}, received base_dir: {base_dir}")

#     effective_base_dir = base_dir
#     if effective_base_dir is None:
#         script_dir_sim_data = os.path.dirname(os.path.abspath(__file__)) 
#         # Adjust based on whether Data_Gen is in src or directly in project root
#         # Assuming Data_Gen is in src, and src is in project_root which contains Data/
#         # path_to_src = os.path.dirname(script_dir_sim_data)
#         # path_to_project_root = os.path.dirname(path_to_src)
#         # effective_base_dir = os.path.join(path_to_project_root, "Data", "C-MAPSS")
#         # For your structure: PROJECT_ROOT/Data_Gen/ and PROJECT_ROOT/Data/
#         path_to_project_root = os.path.dirname(script_dir_sim_data) # if Data_Gen is sibling to Data in PROJECT_ROOT
#         effective_base_dir = os.path.join(path_to_project_root, "Data", "C-MAPSS")

#         print(f"[sim_data.py generate_drifted_engine] Warning: base_dir was None, defaulting to: {effective_base_dir}")
    
#     # print(f"[sim_data.py generate_drifted_engine] Effective base_dir for load_cmaps_data: {effective_base_dir}")
#     # target_train_file = f"train_{dataset_id_for_base}.txt"
#     # full_path_to_check = os.path.join(effective_base_dir, target_train_file)
#     # print(f"[sim_data.py generate_drifted_engine] Full path being checked: {full_path_to_check}")
#     # print(f"[sim_data.py generate_drifted_engine] Does path exist? {os.path.exists(full_path_to_check)}")
#     # print(f"[sim_data.py generate_drifted_engine] Is it a file? {os.path.isfile(full_path_to_check)}")

#     if not os.path.exists(os.path.join(effective_base_dir, f"train_{dataset_id_for_base}.txt")):
#         raise FileNotFoundError(f"CRITICAL in generate_drifted_engine: File not found at calculated path: {os.path.join(effective_base_dir, f'train_{dataset_id_for_base}.txt')}")

#     temp_train_df, _, _ = load_cmaps_data(base_dir=effective_base_dir, dataset_id=dataset_id_for_base)
    
#     unit_df = temp_train_df[temp_train_df["unit"] == base_unit_id].copy()
#     if unit_df.empty:
#         raise ValueError(f"Base unit ID {base_unit_id} not found in dataset {dataset_id_for_base} loaded from {effective_base_dir}.")

#     unit_df = add_rul_column(unit_df) # RUL will be forced later in run_experiment.py for the demo
#     unit_df_processed = discretize_data(unit_df, config=MANUAL_BINS, strategy=BINNING_STRATEGY, n_bins=N_BINS)
#     df_drifted = unit_df_processed.copy()

#     if scenario == "EGT_Drift":
#         print(f"  Applying EGT_Drift to unit {base_unit_id}...")
#         drift_sensor_disc = "sensor_11_disc" # Example EGT-related discretized sensor
        
#         if drift_sensor_disc not in df_drifted.columns:
#             print(f"  Warning: Drift sensor '{drift_sensor_disc}' not found in columns: {df_drifted.columns}. Skipping drift.")
#             return df_drifted

#         n_points = len(df_drifted)
#         if n_points == 0: 
#             print("  Warning: Base unit for drift has no data points. Returning empty DataFrame.")
#             return df_drifted
            
#         max_bin_value = N_BINS - 1
        
#         # --- MODIFIED AGGRESSIVE DRIFT ---
#         # Create a drift pattern that goes from bin 0 to max_bin_value linearly over n_points
#         if n_points > 1:
#             drift_values_pattern = np.linspace(0, max_bin_value, n_points).astype(int)
#         else: # Handle single point case
#             drift_values_pattern = np.array([0]).astype(int) 
            
#         df_drifted[drift_sensor_disc] = np.clip(drift_values_pattern, 0, max_bin_value)
#         # --- END MODIFIED AGGRESSIVE DRIFT ---
        
#         print(f"  Applied AGGRESSIVE drift to '{drift_sensor_disc}' from bin 0 towards {max_bin_value}.")
#         # print(f"  Drifted {drift_sensor_disc} values (first 10): {df_drifted[drift_sensor_disc].head(10).values}")
#         # print(f"  Drifted {drift_sensor_disc} values (last 10): {df_drifted[drift_sensor_disc].tail(10).values}")


#     elif scenario == "Vibration_Spike":
#         # ... (Your Vibration_Spike logic - ensure it uses N_BINS for max_bin_value) ...
#         print(f"  Applying Vibration_Spike to unit {base_unit_id}...")
#         spike_sensor_disc = "sensor_4_disc"
#         if spike_sensor_disc not in df_drifted.columns: 
#             print(f"  Warning: Spike sensor '{spike_sensor_disc}' not found. Skipping spike.")
#             return df_drifted
        
#         n_points = len(df_drifted)
#         if n_points > 10 : 
#             spike_start_index = n_points // 2
#             spike_end_index = spike_start_index + 5 
#             max_bin_value = N_BINS - 1 
#             valid_indices = df_drifted.index[spike_start_index:min(spike_end_index, n_points)]
#             if not valid_indices.empty:
#                 df_drifted.loc[valid_indices, spike_sensor_disc] = max_bin_value
#         # print(f"  Applied spike to '{spike_sensor_disc}'.")
        
#     else:
#         print(f"  Warning: Unknown drift scenario '{scenario}'. No drift applied.")

#     if n_steps is not None:
#         if len(df_drifted) > n_steps:
#             df_drifted = df_drifted.iloc[:n_steps]
#         # Else: Paddington logic would go here if n_steps > len(df_drifted)
    
#     return df_drifted

if __name__ == "__main__":
    # This block is for testing sim_data.py directly.
    # Paths need to be relative to Data_Gen/ or absolute.
    
    # Path from Data_Gen/sim_data.py to PROJECT_ROOT/Data/C-MAPSS is "../../Data/C-MAPSS"
    # Assuming PROJECT_ROOT -> src -> Data_Gen
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # test_base_dir assumes Data_Gen is in src, and Data is sibling to src's parent.
    # If Data_Gen is directly in PROJECT_ROOT, then it's "../Data/C-MAPSS"
    # Your tree: PROJECT_ROOT/Data_Gen and PROJECT_ROOT/Data. So it's "../Data/C-MAPSS"
    test_base_dir = os.path.join(current_file_dir, "..", "Data", "C-MAPSS") # Corrected for your tree
    print(f"Test base_dir for standalone sim_data.py run: {os.path.abspath(test_base_dir)}")


    print("--- Testing prepare_cmaps_data ---")
    try:
        # For standalone test, config.py must be findable.
        # If config.py is in Data_Gen, 'from config import ...' should work.
        from config import OBSERVATION_NODES as obs_nodes_for_test_prepare
        train_df, _, _ = prepare_cmaps_data(base_dir=test_base_dir, drift_scenario="EGT_Drift") 
        print(train_df[['unit', 'cycle', 'sensor_11_disc'] + obs_nodes_for_test_prepare[:2]].head())
    except Exception as e:
        print(f"Error in prepare_cmaps_data test: {e}")


    print("\n--- Testing generate_drifted_engine ---")
    try:
        from config import OBSERVATION_NODES as obs_nodes_for_test_generate
        drifted_unit_data = generate_drifted_engine(scenario="EGT_Drift", n_steps=50, base_unit_id=1, base_dir=test_base_dir)
        print(drifted_unit_data[['unit', 'cycle', 'sensor_11_disc'] + obs_nodes_for_test_generate[:2]].head())
        print(f"Shape of drifted_unit_data: {drifted_unit_data.shape}")
    except Exception as e:
        print(f"Error during generate_drifted_engine EGT_Drift test: {e}")