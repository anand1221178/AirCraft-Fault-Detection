import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.legend
import warnings 
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Import configuration parameters - contains thresholds for discretization and sensor details
from config import PARAMS, DROPOUT_RATE

# --- Helper Function ---
def discretize_value(value, low_thresh, high_thresh):
    """
    Discretize a single sensor reading into 'Low', 'Medium', 'High' categories.
    
    Args:
        value: The raw sensor reading
        low_thresh: Threshold below which value is classified as 'Low'
        high_thresh: Threshold above which value is classified as 'High'
        
    Returns:
        String representing discretized value ('Low', 'Medium', or 'High') or NaN for missing values
    """
    if pd.notna(value):  # Check if the value is not NaN
        if value < low_thresh:
            return 'Low'
        elif value > high_thresh:
            return 'High'
        else:
            return 'Medium'
    else:
        return np.nan  # Preserve NaN values

# --- Main Discretization Function ---
def discretize_data(df, params):
    """
    Discretize all sensor readings in the DataFrame based on parameter thresholds.
    
    Args:
        df: DataFrame with raw sensor readings
        params: Dictionary containing thresholds for each sensor type
        
    Returns:
        DataFrame with original columns plus additional discretized columns
    """
    discretized_df = df.copy()  # Create a copy to avoid modifying the original data
    
    # Loop through each sensor type defined in the parameters
    for sensor, p in params.items():
        # Special handling for vibration sensors (has two channels: Vib1 and Vib2)
        if sensor == 'Vibration':
            discretized_df['Vib1_IPS_Discrete'] = discretized_df['Vib1_IPS'].apply(
                lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
            discretized_df['Vib2_IPS_Discrete'] = discretized_df['Vib2_IPS'].apply(
                lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
        else:
            # For other sensors, create the column name and discretize the values
            col_name = f"{sensor}_{p['unit'].replace('%','Pct')}"
            discretized_df[f"{col_name}_Discrete"] = discretized_df[col_name].apply(
                lambda x: discretize_value(x, p['low_thresh'], p['high_thresh']))
    return discretized_df

# --- Plotting Function ---
def plot_raw_and_discretized(df, sensor, p, scenario):
    """
    Create and save plots showing both raw and discretized sensor data.
    
    Args:
        df: DataFrame containing the data for a specific scenario
        sensor: The sensor type being plotted
        p: Parameters for this sensor (thresholds, units)
        scenario: The fault scenario name for labeling
    """
    fig, ax1 = plt.subplots(figsize=(15, 6))
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