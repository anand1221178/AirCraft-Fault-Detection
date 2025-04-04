import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.legend
import warnings  # To suppress UserWarnings from matplotlib about NaNs
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Import configuration parameters
from config import PARAMS, DROPOUT_RATE

# --- Helper Function ---
def discretize_value(value, low_thresh, high_thresh):
    """Discretize a single sensor reading into 'Low', 'Medium', 'High'."""
    if pd.notna(value):
        if value < low_thresh:
            return 'Low'
        elif value > high_thresh:
            return 'High'
        else:
            return 'Medium'
    else:
        return np.nan

# --- Main Discretization Function ---
def discretize_data(df, params):
    """Discretize sensor readings in the DataFrame."""
    discretized_df = df.copy()
    for sensor, p in params.items():
        if sensor == 'Vibration':
            discretized_df['Vib1_IPS_Discrete'] = discretized_df['Vib1_IPS'].apply(lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
            discretized_df['Vib2_IPS_Discrete'] = discretized_df['Vib2_IPS'].apply(lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
        else:
            col_name = f"{sensor}_{p['unit'].replace('%','Pct')}"
            discretized_df[f"{col_name}_Discrete"] = discretized_df[col_name].apply(lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
    return discretized_df

# --- Plotting Function ---
def plot_raw_and_discretized(df, sensor, p, scenario):
    """Plot raw and discretized sensor data for validation."""
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.set_title(f"Raw and Discretized {sensor} Data - Scenario: {scenario} (Dropout={DROPOUT_RATE*100}%)", fontsize=16)
    
    # Plot raw sensor data
    if sensor == 'Vibration':
        ax1.plot(df['Timestamp'], df['Vib1_IPS'], label='Vib1_IPS (Raw)', alpha=0.7, marker='.', linestyle='-', markersize=1)
        ax1.plot(df['Timestamp'], df['Vib2_IPS'], label='Vib2_IPS (Raw)', alpha=0.7, marker='.', linestyle='-', markersize=1)
        ax1.axhline(p['low_thresh'], color='orange', linestyle=':', label=f'Low Threshold ({p["low_thresh"]})')
        ax1.axhline(p['high_thresh'], color='red', linestyle=':', label=f'High Threshold ({p["high_thresh"]})')
        ax1.set_ylabel(f"Vibration ({p['unit']})")
        ax1.legend(loc='upper left')
        
        # Create a secondary y-axis for discretized data
        ax2 = ax1.twinx()
        # Map discrete values to numerical values for plotting
        discrete_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        ax2.plot(df['Timestamp'], df['Vib1_IPS_Discrete'].map(discrete_mapping), label='Vib1_IPS (Discrete)', color='green', alpha=0.7, marker='x', linestyle='-', markersize=3)
        ax2.plot(df['Timestamp'], df['Vib2_IPS_Discrete'].map(discrete_mapping), label='Vib2_IPS (Discrete)', color='blue', alpha=0.7, marker='x', linestyle='-', markersize=3)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Medium', 'High'])
        ax2.set_ylabel("Discretized Vibration Level")
        ax2.legend(loc='upper right')
    else:
        col_name = f"{sensor}_{p['unit'].replace('%','Pct')}"
        ax1.plot(df['Timestamp'], df[col_name], label=f'{col_name} (Raw)', alpha=0.7, marker='.', linestyle='-', markersize=1)
        ax1.axhline(p['low_thresh'], color='orange', linestyle=':', label=f'Low ({p["low_thresh"]})')
        ax1.axhline(p['high_thresh'], color='red', linestyle=':', label=f'High ({p["high_thresh"]})')
        ax1.set_ylabel(f"{sensor} ({p['unit']})")
        ax1.legend(loc='upper left')
        
        # Create a secondary y-axis for discretized data
        ax2 = ax1.twinx()
        # Map discrete values to numerical values for plotting
        discrete_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        ax2.plot(df['Timestamp'], df[f"{col_name}_Discrete"].map(discrete_mapping), label=f'{col_name} (Discrete)', color='green', alpha=0.7, marker='x', linestyle='-', markersize=3)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Low', 'Medium', 'High'])
        ax2.set_ylabel("Discretized Level")
        ax2.legend(loc='upper right')
    
    ax1.set_xlabel("Timestamp")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap
    plot_filename = os.path.join(data_dir, f'discretized_plot_{sensor}_{scenario}.png')
    plt.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"Discretized plot for {sensor} in scenario {scenario} saved to {plot_filename}")

# --- Example Usage & Validation ---
if __name__ == "__main__":
    # Ensure the Data directory exists
    data_dir = os.path.join(os.path.dirname(__file__), 'Data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Load raw data
    input_filename_raw = os.path.join(data_dir, 'sim_data_raw.csv')
    raw_data = pd.read_csv(input_filename_raw)
    
    # Discretize data
    discretized_data = discretize_data(raw_data, PARAMS)
    
    # Save discretized data
    output_filename_discrete = os.path.join(data_dir, 'sim_data_discrete.csv')
    discretized_data.to_csv(output_filename_discrete, index=False)
    print(f"Discretized data saved to {output_filename_discrete}")
    
    # --- Display Stats ---
    print("\n--- Data Summary ---")
    print(f"Total data points: {len(discretized_data)}")
    print("\nSample Data:")
    print(discretized_data.sample(5))
    print("\n--- Discretization script finished ---")
    
    # --- Plotting Raw and Discretized Data ---
    scenarios_to_plot = ['Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail']
    for scenario in scenarios_to_plot:
        print(f"Plotting raw and discretized data for scenario: {scenario}")
        df_scenario = discretized_data[discretized_data['Scenario'] == scenario]
        for sensor, p in PARAMS.items():
            plot_raw_and_discretized(df_scenario, sensor, p, scenario)