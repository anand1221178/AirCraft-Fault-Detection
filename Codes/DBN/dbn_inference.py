# dbn_inference.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pgmpy.inference import DBNInference 

# Import functions from your previous script
try:
    from dbn_model import define_dbn_structure, define_initial_cpts
except ImportError:
    print("Error: Could not import from dbn_model.py. Make sure it's in the same directory.")
    exit()

# --- Configuration ---
DATA_DIR = '../Data' # Adjusted relative path if needed
DISCRETE_DATA_FILE = os.path.join(DATA_DIR, 'sim_data_discrete.csv')
PLOT_SAVE_DIR = os.path.join(DATA_DIR, 'inference_plots_all_scenarios') # New directory for plots

# --- Mappings ---
OBS_STATE_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
OBSERVATION_VARS = [
    'EGT_C_Discrete', 'N2_PctRPM_Discrete', 'OilPressure_PSI_Discrete',
    'Vib1_IPS_Discrete', 'Vib2_IPS_Discrete'
]
# Query all hidden variables
HIDDEN_VARS_TO_QUERY = [
    'Engine_Core_Health', 
    'Lubrication_System_Health',
    'EGT_Sensor_Health',        
    'Vibration_Sensor_Health'   
]

# State names map for formatting results
state_names = {
    'Engine_Core_Health': ['OK', 'Warn', 'Fail'],
    'Lubrication_System_Health': ['OK', 'Fail'],
    'EGT_Sensor_Health': ['OK', 'Degraded', 'Failed'],     
    'Vibration_Sensor_Health': ['OK', 'Degraded', 'Failed'] 
}

# Simulation parameters (needed for finding onset times)
FAULT_ONSET_FRACTION = 0.4 
SENSOR_FAILURE_ONSET_FRACTION = 0.6

# --- Functions (Keep prepare_evidence_sequence, run_dbn_inference, format_results_to_dataframe as before) ---
def prepare_evidence_sequence(df_slice, obs_vars, state_map):
    """Converts DataFrame slice to evidence sequence."""
    evidence_sequence = []
    num_steps = len(df_slice)
    df_reset = df_slice.reset_index()
    original_start_index = df_slice.index[0] if isinstance(df_slice.index, pd.RangeIndex) else 0

    for t in range(num_steps):
        evidence_dict_t = {}
        row = df_reset.iloc[t]
        current_original_index = row.get('level_0', t + original_start_index)

        for var_base_name in obs_vars:
            observed_value = row.get(var_base_name, np.nan)
            if pd.notna(observed_value):
                try:
                    mapped_state = state_map[observed_value]
                    evidence_dict_t[(var_base_name, t)] = mapped_state
                except KeyError:
                    print(f"Warning: Unknown state '{observed_value}' for {var_base_name} at step {t}. Skipping.")
                except TypeError:
                     print(f"Warning: Non-string state '{observed_value}' ({type(observed_value)}) for {var_base_name} at step {t}. Skipping.")
        evidence_sequence.append(evidence_dict_t)
    return evidence_sequence

def run_dbn_inference(dbn_model, evidence_sequence, hidden_vars_base):
    """Runs DBNInference forward inference."""
    print("Initializing DBNInference...")
    inference_engine = DBNInference(dbn_model)
    results_sequence = []
    num_steps = len(evidence_sequence)
    print(f"Running forward inference (filtering) for {num_steps} time steps...")

    for t in range(num_steps):
        if (t + 1) % 50 == 0 or t == num_steps - 1:
             print(f"  Processing time step {t+1}/{num_steps}")
        current_evidence = evidence_sequence[t]
        query_vars_t = [(var_base, t) for var_base in hidden_vars_base]
        try:
            marginals_at_t = inference_engine.forward_inference(
                variables=query_vars_t,
                evidence=current_evidence
            )
            results_sequence.append(marginals_at_t)
        except Exception as e:
             print(f"\nError during inference at step {t}: {e}")
             print(f"Current evidence: {current_evidence}")
             print(f"Query variables: {query_vars_t}")
             # Optionally re-raise or append empty results
             results_sequence.append({}) # Append empty dict to maintain sequence length
             # raise e # Or re-raise the exception to stop execution
    print("Inference complete.")
    return results_sequence

def format_results_to_dataframe(results_sequence, hidden_vars_base, state_names_map):
    """Converts inference results to a DataFrame."""
    num_steps = len(results_sequence)
    data_for_df = {'TimeStep': list(range(num_steps))}

    for base_var in hidden_vars_base:
        try:
            states = state_names_map[base_var]
            for state_name in states:
                col_name = f"P({base_var}={state_name})"
                data_for_df[col_name] = [np.nan] * num_steps
        except KeyError:
            print(f"Warning: State names not provided for '{base_var}'. Skipping.")
            continue

    for t, marginals_dict in enumerate(results_sequence):
        # Skip if inference failed at this step and dict is empty
        if not marginals_dict:
            continue
        for base_var in hidden_vars_base:
            if base_var not in state_names_map: continue
            temporal_var = (base_var, t)
            if temporal_var in marginals_dict:
                factor = marginals_dict[temporal_var]
                probabilities = factor.values
                num_states = len(state_names_map[base_var])
                if len(probabilities) != num_states:
                    print(f"Warning: State/probability mismatch for {temporal_var} at {t}.")
                    continue
                for i in range(num_states):
                    state_name = state_names_map[base_var][i]
                    col_name = f"P({base_var}={state_name})"
                    data_for_df[col_name][t] = probabilities[i]
            else:
                 print(f"Warning: No marginal for {temporal_var} at {t}")


    df = pd.DataFrame(data_for_df)
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Initializing DBN Model ---")
    try:
        dbn = define_dbn_structure()
        cpt_list = define_initial_cpts()
        print(f"Defined DBN structure and {len(cpt_list)} CPTs.")
        for cpt in cpt_list:
            dbn.add_cpds(cpt)
        print("CPTs added to DBN.")
        dbn.check_model()
        print("DBN model check passed.")
    except Exception as e:
        print(f"Error initializing DBN: {e}")
        exit()

    print(f"\n--- Loading Discretized Data: {DISCRETE_DATA_FILE} ---")
    try:
        df_discrete = pd.read_csv(DISCRETE_DATA_FILE, parse_dates=['Timestamp'])
        print(f"Loaded data shape: {df_discrete.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {DISCRETE_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- List of Scenarios to Test ---
    scenarios_to_test = ['Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail']

    # Ensure plot directory exists
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    # --- Loop Through Scenarios ---
    for scenario_name in scenarios_to_test:
        print(f"\n{'='*15} Processing Scenario: {scenario_name} {'='*15}")

        print("\n--- Preparing Test Sequence ---")
        try:
            scenario_df = df_discrete[df_discrete['Scenario'] == scenario_name].copy()
            if scenario_df.empty:
                 print(f"Warning: No data found for scenario '{scenario_name}'. Skipping.")
                 continue

            # Select a slice (e.g., first 100 steps, or around onset)
            # Let's take a slice around the relevant onset time for faults/failures
            # Or just a fixed slice for 'Normal'
            slice_len = 100 # Define desired slice length
            onset_fraction = -1
            if scenario_name in ['OilLeak', 'BearingWear']:
                onset_fraction = FAULT_ONSET_FRACTION
            elif scenario_name in ['EGTSensorFail', 'VibSensorFail']:
                 onset_fraction = SENSOR_FAILURE_ONSET_FRACTION

            if onset_fraction > 0:
                onset_index_abs = scenario_df.index[int(len(scenario_df) * onset_fraction)]
                onset_timestamp = df_discrete.loc[onset_index_abs, 'Timestamp']
                start_time = onset_timestamp - pd.Timedelta(seconds=slice_len // 2)
                end_time = onset_timestamp + pd.Timedelta(seconds=slice_len // 2)
                # Filter original df to get the slice
                test_slice_df = df_discrete[
                    (df_discrete['Timestamp'] >= start_time) &
                    (df_discrete['Timestamp'] <= end_time) &
                    (df_discrete['Scenario'] == scenario_name)
                ].copy()
                # Calculate relative onset step
                onset_row = test_slice_df[test_slice_df['Timestamp'] >= onset_timestamp]
                event_onset_relative_step = onset_row.index[0] - test_slice_df.index[0] if not onset_row.empty else -1
                event_label = "Fault Onset" if onset_fraction == FAULT_ONSET_FRACTION else "Sensor Failure Onset"

            else: # Normal scenario, just take first steps
                test_slice_df = scenario_df.head(slice_len).copy()
                event_onset_relative_step = -1 # No specific event
                event_label = "N/A"


            if test_slice_df.empty:
                 print(f"Warning: Selected time slice for '{scenario_name}' is empty. Skipping.")
                 continue

            test_slice_df.sort_values(by='Timestamp', inplace=True)
            test_slice_df.reset_index(inplace=True, drop=True) # Use 0-based index

            print(f"Selected {len(test_slice_df)} time steps for '{scenario_name}'.")

        except Exception as e:
            print(f"Error selecting data slice for {scenario_name}: {e}")
            continue # Skip to next scenario

        # Prepare evidence
        evidence_sequence = prepare_evidence_sequence(test_slice_df, OBSERVATION_VARS, OBS_STATE_MAP)
        if not evidence_sequence:
            print(f"Error: No evidence generated for {scenario_name}. Skipping.")
            continue

        # --- Run Inference ---
        print("\n--- Running DBN Inference ---")
        try:
            inference_results = run_dbn_inference(dbn, evidence_sequence, HIDDEN_VARS_TO_QUERY)
        except Exception as e:
            print(f"Error during inference for {scenario_name}: {e}")
            continue # Skip to next scenario

        # --- Process and Display Results ---
        print("\n--- Processing Inference Results ---")
        results_df = format_results_to_dataframe(inference_results, HIDDEN_VARS_TO_QUERY, state_names)
        print(f"Sample Inference Results ({scenario_name}):")
        print(results_df.head(3))
        print(results_df.tail(3))

        # --- Plotting ---
        print("\n--- Plotting Inference Results ---")
        num_hidden_vars = len(HIDDEN_VARS_TO_QUERY)
        fig, axes = plt.subplots(num_hidden_vars, 1, figsize=(15, 5 * num_hidden_vars), sharex=True)
        fig.suptitle(f'DBN Inference Results: Scenario = {scenario_name}')

        if num_hidden_vars == 1: axes = [axes] # Make axes iterable if only one subplot

        for i, base_var in enumerate(HIDDEN_VARS_TO_QUERY):
            ax = axes[i]
            # Get relevant columns safely
            state_list = state_names.get(base_var, [])
            plot_cols_exist = [f"P({base_var}={state})" for state in state_list if f"P({base_var}={state})" in results_df.columns]

            if not plot_cols_exist:
                print(f"Warning: No result columns found for {base_var} in {scenario_name}. Skipping plot.")
                continue

            results_df.plot(y=plot_cols_exist, ax=ax, marker='.', linestyle='-') # Use default index
            ax.set_ylabel('Probability')
            ax.set_title(f'Inferred Probabilities for {base_var}')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) # Move legend out

            # Add event onset line if applicable
            if event_onset_relative_step != -1:
                 ax.axvline(event_onset_relative_step, color='r', linestyle='--', label=event_label)
                 # Re-call legend to potentially include vline (might need better handling)
                 ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        axes[-1].set_xlabel('Time Step in Slice')
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95]) # Adjust layout for legend
        plot_filename = os.path.join(PLOT_SAVE_DIR, f'dbn_inference_{scenario_name}.png')
        plt.savefig(plot_filename, dpi=150)
        plt.close(fig) # Close figure to prevent display and save memory
        print(f"Saved inference plot to {plot_filename}")

    print("\n--- DBN Inference Script Finished ---")