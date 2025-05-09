# dbn_inference.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pgmpy.inference import DBNInference 

# Import functions from dbn_model.py
try:
    from dbn_model import define_dbn_structure, define_initial_cpts
except ImportError:
    print("Error: Could not import from dbn_model.py. Make sure it's in the same directory.")
    exit()

# --- Configuration ---
DATA_DIR = '../Data' 
DISCRETE_DATA_FILE = os.path.join(DATA_DIR, 'sim_data_discrete.csv')
PLOT_SAVE_DIR = os.path.join(DATA_DIR, 'inference_plots_all_scenarios') # New directory for plots

# --- Mappings ---
OBS_STATE_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
OBSERVATION_VARS = [
    'EGT_C_Discrete', 'N2_PctRPM_Discrete', 'OilPressure_PSI_Discrete',
    'Vib1_IPS_Discrete', 'Vib2_IPS_Discrete'
]

# Query all hidden variables for inference
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

# --- Functions ---
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

# Run DBN Inference 
def run_dbn_inference(dbn_model, evidence_sequence, hidden_vars_base):

    """Runs DBNInference forward inference."""

    # Initialize the DBNInference engine with the given DBN model
    print("Initializing DBNInference...")
    inference_engine = DBNInference(dbn_model) 
    results_sequence = []  # List to store inference results for each time step
    num_steps = len(evidence_sequence)  # Total number of time steps in the evidence sequence
    print(f"Running forward inference (filtering) for {num_steps} time steps...")

    # Loop through each time step in the evidence sequence
    for t in range(num_steps):
        # Print progress every 50 steps or at the last step
        if (t + 1) % 50 == 0 or t == num_steps - 1:
             print(f"  Processing time step {t+1}/{num_steps}")
        
        # Extract the evidence for the current time step
        current_evidence = evidence_sequence[t]
        
        # Prepare the query variables for the current time step
        # These are the hidden variables at the current time step
        query_vars_t = [(var_base, t) for var_base in hidden_vars_base]
        
        try:
            # Perform forward inference for the current time step
            marginals_at_t = inference_engine.forward_inference(
                variables=query_vars_t,
                evidence=current_evidence
            )
            # Append the inference results (marginals) to the results list
            results_sequence.append(marginals_at_t)
        except Exception as e:
            # Handle any errors during inference
            print(f"\nError during inference at step {t}: {e}")
            print(f"Current evidence: {current_evidence}")  # Log the evidence causing the error
            print(f"Query variables: {query_vars_t}")  # Log the query variables causing the error
            results_sequence.append({})  # Append an empty result for the failed step

    
    print("Inference complete.")
    return results_sequence  # Return the sequence of inference results

def format_results_to_dataframe(results_sequence, hidden_vars_base, state_names_map):
    """Converts inference results to a DataFrame."""
    
    # Get the total number of time steps in the results sequence
    num_steps = len(results_sequence)
    
    # Initialize a dictionary to store data for the DataFrame
    # Start with a 'TimeStep' column containing the time step indices
    data_for_df = {'TimeStep': list(range(num_steps))}

    # Loop through each hidden variable to query
    for base_var in hidden_vars_base:
        try:
            # Get the possible states for the current variable from the state names map
            states = state_names_map[base_var]
            for state_name in states:
                # Create a column for each state probability and initialize with NaN
                col_name = f"P({base_var}={state_name})"
                data_for_df[col_name] = [np.nan] * num_steps
        except KeyError:
            # Handle cases where the state names for the variable are not provided
            print(f"Warning: State names not provided for '{base_var}'. Skipping.")
            continue

    # Loop through each time step and its corresponding inference results
    for t, marginals_dict in enumerate(results_sequence):
        # Skip this time step if inference failed and the result is empty
        if not marginals_dict:
            continue
        
        # Loop through each hidden variable to query
        for base_var in hidden_vars_base:
            # Skip if the variable is not in the state names map
            if base_var not in state_names_map:
                continue
            
            # Create the temporal variable for the current time step
            temporal_var = (base_var, t)
            
            # Check if the marginal distribution for the variable exists in the results
            if temporal_var in marginals_dict:
                # Extract the marginal distribution (factor) for the variable
                factor = marginals_dict[temporal_var]
                probabilities = factor.values  # Get the probability values
                
                # Ensure the number of probabilities matches the number of states
                num_states = len(state_names_map[base_var])
                if len(probabilities) != num_states:
                    print(f"Warning: State/probability mismatch for {temporal_var} at {t}.")
                    continue
                
                # Assign the probabilities to the corresponding columns in the DataFrame
                for i in range(num_states):
                    state_name = state_names_map[base_var][i]
                    col_name = f"P({base_var}={state_name})"
                    data_for_df[col_name][t] = probabilities[i]
            else:
                # Handle cases where no marginal distribution is found for the variable
                print(f"Warning: No marginal for {temporal_var} at {t}")

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(data_for_df)
    return df

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Initialize the DBN Model ---
    print("--- Initializing DBN Model ---")
    try:
        # Define the DBN structure and conditional probability tables (CPTs)
        dbn = define_dbn_structure()
        cpt_list = define_initial_cpts()
        print(f"Defined DBN structure and {len(cpt_list)} CPTs.")
        
        # Add CPTs to the DBN model
        for cpt in cpt_list:
            dbn.add_cpds(cpt)
        print("CPTs added to DBN.")
        
        # Check if the DBN model is valid
        dbn.check_model()
        print("DBN model check passed.")
    except Exception as e:
        # Handle errors during DBN initialization
        print(f"Error initializing DBN: {e}")
        exit()

    # --- Load Discretized Data ---
    print(f"\n--- Loading Discretized Data: {DISCRETE_DATA_FILE} ---")
    try:
        # Load the discretized data from the specified file
        df_discrete = pd.read_csv(DISCRETE_DATA_FILE, parse_dates=['Timestamp'])
        print(f"Loaded data shape: {df_discrete.shape}")
    except FileNotFoundError:
        # Handle file not found error
        print(f"Error: File not found at {DISCRETE_DATA_FILE}")
        exit()
    except Exception as e:
        # Handle other errors during data loading
        print(f"Error loading data: {e}")
        exit()

    # --- List of Scenarios to Test ---
    scenarios_to_test = ['Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail']

    # Ensure the directory for saving plots exists
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    # --- Loop Through Each Scenario ---
    for scenario_name in scenarios_to_test:
        print(f"\n{'='*15} Processing Scenario: {scenario_name} {'='*15}")

        print("\n--- Preparing Test Sequence ---")
        try:
            # Filter data for the current scenario
            scenario_df = df_discrete[df_discrete['Scenario'] == scenario_name].copy()
            if scenario_df.empty:
                print(f"Warning: No data found for scenario '{scenario_name}'. Skipping.")
                continue

            # Define the slice length and determine the onset fraction
            slice_len = 100  # Number of time steps to include in the slice
            onset_fraction = -1
            if scenario_name in ['OilLeak', 'BearingWear']:
                onset_fraction = FAULT_ONSET_FRACTION
            elif scenario_name in ['EGTSensorFail', 'VibSensorFail']:
                onset_fraction = SENSOR_FAILURE_ONSET_FRACTION

            if onset_fraction > 0:
                # Calculate the onset time and define the slice around it
                onset_index_abs = scenario_df.index[int(len(scenario_df) * onset_fraction)]
                onset_timestamp = df_discrete.loc[onset_index_abs, 'Timestamp']
                start_time = onset_timestamp - pd.Timedelta(seconds=slice_len // 2)
                end_time = onset_timestamp + pd.Timedelta(seconds=slice_len // 2)
                
                # Filter the data slice for the current scenario
                test_slice_df = df_discrete[
                    (df_discrete['Timestamp'] >= start_time) &
                    (df_discrete['Timestamp'] <= end_time) &
                    (df_discrete['Scenario'] == scenario_name)
                ].copy()
                
                # Calculate the relative onset step
                onset_row = test_slice_df[test_slice_df['Timestamp'] >= onset_timestamp]
                event_onset_relative_step = onset_row.index[0] - test_slice_df.index[0] if not onset_row.empty else -1
                event_label = "Fault Onset" if onset_fraction == FAULT_ONSET_FRACTION else "Sensor Failure Onset"
            else:
                # For normal scenarios, take the first `slice_len` steps
                test_slice_df = scenario_df.head(slice_len).copy()
                event_onset_relative_step = -1  # No specific event
                event_label = "N/A"

            if test_slice_df.empty:
                # Handle empty slices
                print(f"Warning: Selected time slice for '{scenario_name}' is empty. Skipping.")
                continue

            # Sort and reset the index for the selected slice
            test_slice_df.sort_values(by='Timestamp', inplace=True)
            test_slice_df.reset_index(inplace=True, drop=True)

            print(f"Selected {len(test_slice_df)} time steps for '{scenario_name}'.")

        except Exception as e:
            # Handle errors during data slicing
            print(f"Error selecting data slice for {scenario_name}: {e}")
            continue  # Skip to the next scenario

        # --- Prepare Evidence ---
        evidence_sequence = prepare_evidence_sequence(test_slice_df, OBSERVATION_VARS, OBS_STATE_MAP)
        if not evidence_sequence:
            # Handle cases where no evidence is generated
            print(f"Error: No evidence generated for {scenario_name}. Skipping.")
            continue

        # --- Run DBN Inference ---
        print("\n--- Running DBN Inference ---")
        try:
            # Perform inference using the DBN model
            inference_results = run_dbn_inference(dbn, evidence_sequence, HIDDEN_VARS_TO_QUERY)
        except Exception as e:
            # Handle errors during inference
            print(f"Error during inference for {scenario_name}: {e}")
            continue  # Skip to the next scenario

        # --- Process and Display Results ---
        print("\n--- Processing Inference Results ---")
        results_df = format_results_to_dataframe(inference_results, HIDDEN_VARS_TO_QUERY, state_names)
        print(f"Sample Inference Results ({scenario_name}):")
        print(results_df.head(3))  # Display the first 3 rows of results
        print(results_df.tail(3))  # Display the last 3 rows of results

        # --- Plotting ---
        print("\n--- Plotting Inference Results ---")
        num_hidden_vars = len(HIDDEN_VARS_TO_QUERY)
        fig, axes = plt.subplots(num_hidden_vars, 1, figsize=(15, 5 * num_hidden_vars), sharex=True)
        fig.suptitle(f'DBN Inference Results: Scenario = {scenario_name}')

        if num_hidden_vars == 1:
            axes = [axes]  # Ensure axes is iterable if there's only one subplot

        for i, base_var in enumerate(HIDDEN_VARS_TO_QUERY):
            ax = axes[i]
            # Get the relevant columns for the current variable
            state_list = state_names.get(base_var, [])
            plot_cols_exist = [f"P({base_var}={state})" for state in state_list if f"P({base_var}={state})" in results_df.columns]

            if not plot_cols_exist:
                # Handle cases where no result columns exist for the variable
                print(f"Warning: No result columns found for {base_var} in {scenario_name}. Skipping plot.")
                continue

            # Plot the probabilities for the current variable
            results_df.plot(y=plot_cols_exist, ax=ax, marker='.', linestyle='-')
            ax.set_ylabel('Probability')
            ax.set_title(f'Inferred Probabilities for {base_var}')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend outside the plot

            # Add a vertical line for the event onset if applicable
            if event_onset_relative_step != -1:
                ax.axvline(event_onset_relative_step, color='r', linestyle='--', label=event_label)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Update legend to include the line

        # Set the x-axis label for the last subplot
        axes[-1].set_xlabel('Time Step in Slice')
        plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # Adjust layout for the legend
        plot_filename = os.path.join(PLOT_SAVE_DIR, f'dbn_inference_{scenario_name}.png')
        plt.savefig(plot_filename, dpi=150)  # Save the plot to a file
        plt.close(fig)  # Close the figure to save memory
        print(f"Saved inference plot to {plot_filename}")

    # --- Script Completion ---
    print("\n--- DBN Inference Script Finished ---")