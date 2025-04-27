# simulate_data.py

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.legend
import warnings  # To suppress UserWarnings from matplotlib about NaNs
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Import configuration parameters
from config import (
    SIMULATION_DURATION_MINUTES,
    DATA_FREQUENCY_HZ,
    FAULT_ONSET_FRACTION,
    FAULT_PROGRESSION_FRACTION,
    SENSOR_FAILURE_ONSET_FRACTION,
    SENSOR_FAILURE_PROGRESSION_FRACTION,
    DEFAULT_NOISE_STD_FACTOR,
    DROPOUT_RATE,
    PARAMS
)

# --- Helper Function ---
def smooth_step(x):
    """Smooth transition from 0 to 1 as x goes from 0 to 1."""
    x = np.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3

# --- Main Simulation Function ---
def simulate_engine_data(duration_minutes, frequency_hz, scenario='Normal', params=PARAMS, dropout_rate=DROPOUT_RATE):
    """
    Generates simulated data for all sensors based on the specified scenario.
    """
    total_seconds = duration_minutes * 60
    num_data_points = total_seconds * frequency_hz
    time_delta = datetime.timedelta(seconds=1.0 / frequency_hz)
    start_time = datetime.datetime.now() - datetime.timedelta(seconds=total_seconds)
    timestamps = [start_time + i * time_delta for i in range(num_data_points)]
    
    # Timing indices
    fault_onset_idx = int(num_data_points * FAULT_ONSET_FRACTION)
    fault_end_progression_idx = fault_onset_idx + int(num_data_points * FAULT_PROGRESSION_FRACTION)
    sensor_fail_onset_idx = int(num_data_points * SENSOR_FAILURE_ONSET_FRACTION)
    sensor_fail_end_progression_idx = sensor_fail_onset_idx + int(num_data_points * SENSOR_FAILURE_PROGRESSION_FRACTION)
    
    # Initialize data dictionary and current state values
    data = {'Timestamp': timestamps}
    current_values = {
        'EGT': params['EGT']['cruise_target'],
        'N2': params['N2']['cruise_target'],
        'OilPressure': params['OilPressure']['cruise_target'],
        'Vibration': params['Vibration']['cruise_target']  # Base 'true' vibration
    }
    
    # Initialize columns for reported values
    for sensor in ['EGT', 'N2', 'OilPressure', 'Vibration']:  # Base names
        if sensor == 'Vibration':
            data['Vib1_IPS'] = np.zeros(num_data_points)
            data['Vib2_IPS'] = np.zeros(num_data_points)
        else:
            col_name = f"{sensor}_{params[sensor]['unit'].replace('%','Pct')}"  # Make col name safe
            data[col_name] = np.zeros(num_data_points)
    
    # Ground Truth Labels
    data['Engine_Fault_State'] = ['Normal'] * num_data_points
    data['EGT_Sensor_Health'] = ['OK'] * num_data_points
    data['Vibration_Sensor_Health'] = ['OK'] * num_data_points  # Represents health of the vibration sensing system/pair
    
    # --- Simulation Loop ---
    for i in range(num_data_points):
        # 1. Determine Fault/Failure Progression
        fault_progress = 0.0
        if scenario in ['OilLeak', 'BearingWear'] and i >= fault_onset_idx:
            fault_progress = smooth_step((i - fault_onset_idx) / max(1, fault_end_progression_idx - fault_onset_idx))
            data['Engine_Fault_State'][i] = f"{scenario}_Active"
        
        sensor_fail_progress = 0.0
        sensor_failing = 'None'
        if scenario == 'EGTSensorFail' and i >= sensor_fail_onset_idx:
            sensor_fail_progress = smooth_step((i - sensor_fail_onset_idx) / max(1, sensor_fail_end_progression_idx - sensor_fail_onset_idx))
            sensor_failing = 'EGT'
        elif scenario == 'VibSensorFail' and i >= sensor_fail_onset_idx:
            sensor_fail_progress = smooth_step((i - sensor_fail_onset_idx) / max(1, sensor_fail_end_progression_idx - sensor_fail_onset_idx))
            sensor_failing = 'Vibration'
        
        # --- 2. Calculate Target TRUE Values for this step ---
        targets = {}
        targets['N2'] = params['N2']['cruise_target']
        if scenario == 'BearingWear':
            targets['N2'] -= fault_progress * params['N2']['bearing_wear_decrease']
        targets['EGT'] = params['EGT']['cruise_target']
        if scenario == 'BearingWear':
            targets['EGT'] += fault_progress * params['EGT']['bearing_wear_increase']
        # EGT also weakly depends on N2, but N2 change is minimal here
        targets['OilPressure'] = params['OilPressure']['cruise_target']
        if scenario == 'OilLeak':
            leak_target = params['OilPressure']['oil_leak_target']
            cruise_target = params['OilPressure']['cruise_target']
            targets['OilPressure'] = cruise_target + fault_progress * (leak_target - cruise_target)
        targets['Vibration'] = params['Vibration']['cruise_target']
        if scenario == 'BearingWear':
            targets['Vibration'] += fault_progress * params['Vibration']['bearing_wear_increase']
        # Vibration weakly depends on N2, minimal effect here
        
        # --- 3. Update Current TRUE Values (Simulate Dynamics/Inertia) ---
        for sensor in ['EGT', 'OilPressure', 'Vibration']:  # N2 changes instantly in this simple model
            p = params[sensor]
            ramp = p.get('ramp_factor', 0.5)  # Default to faster change if no ramp defined
            current_values[sensor] += (targets[sensor] - current_values[sensor]) * ramp
        
        # N2: Update directly based on target (no ramp needed for this model)
        current_values['N2'] = targets['N2']  # Or add a ramp if desired
        
        # --- 4. Simulate Sensor Reading Process (Noise, Failure, Correlation for Vib) ---
        reported_values = {}
        for sensor, p in params.items():
            true_val = current_values[sensor]
            base_noise_std = p.get('noise_std', true_val * p.get('noise_std_factor', DEFAULT_NOISE_STD_FACTOR))
            sensor_health = 'OK'
            reported_val = true_val  # Start with the true value
            
            # Apply sensor failure effects (drift, increased noise)
            if sensor_failing == sensor and p['has_hmm']:
                drift = sensor_fail_progress * p['fail_drift']
                noise_factor = 1.0 + sensor_fail_progress * (p['fail_noise_factor'] - 1.0)
                base_noise_std *= noise_factor
                reported_val += drift  # Add drift *before* noise
                # Update sensor health ground truth
                if sensor_fail_progress < 0.1:
                    sensor_health = 'OK'
                elif sensor_fail_progress < 0.9:
                    sensor_health = 'Degraded'
                else:
                    sensor_health = 'Failed'
                if sensor == 'EGT':
                    data['EGT_Sensor_Health'][i] = sensor_health
                if sensor == 'Vibration':
                    data['Vibration_Sensor_Health'][i] = sensor_health
            
            # Add baseline noise
            noise = np.random.normal(loc=0.0, scale=base_noise_std)
            reported_val += noise
            
            # Handle Vibration specifically (Vib1, Vib2, MRF noise)
            if sensor == 'Vibration':
                # Add small independent noise component for MRF
                noise_vib1 = np.random.normal(loc=0.0, scale=p['mrf_correlation_noise_std'])
                noise_vib2 = np.random.normal(loc=0.0, scale=p['mrf_correlation_noise_std'])
                # Clamp and store Vib1, Vib2
                vib1_final = np.clip(reported_val + noise_vib1, p['min_val'], p['max_val'])
                vib2_final = np.clip(reported_val + noise_vib2, p['min_val'], p['max_val'])
                # Apply dropout individually
                if np.random.rand() < dropout_rate:
                    vib1_final = np.nan
                if np.random.rand() < dropout_rate:
                    vib2_final = np.nan
                data['Vib1_IPS'][i] = vib1_final
                data['Vib2_IPS'][i] = vib2_final
            else:
                # Clamp other sensors
                reported_val_final = np.clip(reported_val, p['min_val'], p['max_val'])
                # Apply dropout
                if np.random.rand() < dropout_rate:
                    reported_val_final = np.nan
                # Store in data dictionary
                col_name = f"{sensor}_{p['unit'].replace('%','Pct')}"
                data[col_name][i] = reported_val_final
    
    # --- End Loop ---
    df = pd.DataFrame(data)
    
    df['Scenario'] = scenario
    # Round numeric columns for realism
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].round(2)
    
    return df

# --- Example Usage & Validation ---
if __name__ == "__main__":
    scenarios_to_run = ['Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail']
    all_scenario_data = []  # Store dataframes for final concatenation
    print("Generating simulated engine data...")
    for scenario in scenarios_to_run:
        print(f"--- Scenario: {scenario} ---")
        df_scenario = simulate_engine_data(
            duration_minutes=SIMULATION_DURATION_MINUTES,
            frequency_hz=DATA_FREQUENCY_HZ,
            scenario=scenario
        )
        df_scenario['Scenario'] = scenario  # Label for combined file/analysis
        all_scenario_data.append(df_scenario)
        
        # --- Plotting for Validation ---
        print(f"    Plotting {scenario}...")
        num_sensors = len(PARAMS) + 1  # +1 because Vib has two outputs
        fig, axes = plt.subplots(num_sensors, 1, figsize=(15, 4 * num_sensors), sharex=True)
        fig.suptitle(f"Simulated Sensor Data - Scenario: {scenario} (Dropout={DROPOUT_RATE*100}%)", fontsize=16)
        plot_idx = 0
        # Plot each sensor
        for sensor, p in PARAMS.items():
            ax = axes[plot_idx]
            if sensor == 'Vibration':
                ax.plot(df_scenario['Timestamp'], df_scenario['Vib1_IPS'], label='Vib1_IPS', alpha=0.7, marker='.', linestyle='-', markersize=1)
                ax.plot(df_scenario['Timestamp'], df_scenario['Vib2_IPS'], label='Vib2_IPS', alpha=0.7, marker='.', linestyle='-', markersize=1)
                ax.axhline(p['low_thresh'], color='orange', linestyle=':', label=f'Low Threshold ({p["low_thresh"]})')
                ax.axhline(p['high_thresh'], color='red', linestyle=':', label=f'High Threshold ({p["high_thresh"]})')
                ax.set_ylabel(f"Vibration ({p['unit']})")
                ax.legend(loc='upper left')
                plot_idx += 1  # Vib uses one plot slot, but we need another one below
                # Need an extra plot slot since Vib has Vib1/Vib2
                ax = axes[plot_idx]
                # Plot Sensor Health if applicable
                if p['has_hmm']:
                    health_map = {'OK': 1, 'Degraded': 0.5, 'Failed': 0}
                    health_numeric = df_scenario['Vibration_Sensor_Health'].map(health_map)
                    ax.plot(df_scenario['Timestamp'], health_numeric, label='Vibration Sensor Health (GT)', color='purple', drawstyle='steps-post')
                    ax.set_yticks([0, 0.5, 1])
                    ax.set_yticklabels(['Failed', 'Degraded', 'OK'])
                    ax.set_ylabel("Sensor Health")
                    ax.legend(loc='upper left')
            else:
                col_name = f"{sensor}_{p['unit'].replace('%','Pct')}"
                ax.plot(df_scenario['Timestamp'], df_scenario[col_name], label=col_name, marker='.', linestyle='-', markersize=1)
                ax.axhline(p['low_thresh'], color='orange', linestyle=':', label=f'Low ({p["low_thresh"]})')
                ax.axhline(p['high_thresh'], color='red', linestyle=':', label=f'High ({p["high_thresh"]})')
                ax.set_ylabel(f"{sensor} ({p['unit']})")
                # Add sensor health subplot if applicable
                if p['has_hmm']:
                    ax_health = ax.twinx()  # Share x-axis
                    health_map = {'OK': 1, 'Degraded': 0.5, 'Failed': 0}
                    health_numeric = df_scenario[f'{sensor}_Sensor_Health'].map(health_map)
                    ax_health.plot(df_scenario['Timestamp'], health_numeric, label=f'{sensor} Sensor Health (GT)', color='purple', drawstyle='steps-post', alpha=0.6)
                    ax_health.set_yticks([0, 0.5, 1])
                    ax_health.set_yticklabels(['Failed', 'Degraded', 'OK'])
                    ax_health.set_ylabel("Sensor Health", color='purple')
                    ax_health.tick_params(axis='y', labelcolor='purple')
                    # Combine legends
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax_health.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc='upper left')
                else:
                    ax.legend(loc='upper left')
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            plot_idx += 1
        
        # Add vertical lines for events
        fault_onset_time = df_scenario['Timestamp'].iloc[int(len(df_scenario) * FAULT_ONSET_FRACTION)]
        sensor_fail_onset_time = df_scenario['Timestamp'].iloc[int(len(df_scenario) * SENSOR_FAILURE_ONSET_FRACTION)]
        # Make legend visible just once on the first axes
        if ax == axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            # Add handles/labels from the twin axis if it exists
            if ax.get_legend() and ax.get_legend().axes != ax:  # Check if twinx exists and has legend items
                twin_handles, twin_labels = ax.get_legend().axes.get_legend_handles_labels()
                handles.extend(twin_handles)
                labels.extend(twin_labels)
            # Remove duplicate labels/handles (like event lines potentially added multiple times)
            by_label = dict(zip(labels, handles))
            # Set the final, unique legend on the first axis
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        elif ax.get_legend():  # Remove legends from other axes if they exist
            ax.get_legend().remove()
        axes[-1].set_xlabel("Timestamp")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap
        # Save the plot
        plt.savefig(f'simulation_plot_{scenario}.png', dpi=150)
        plt.close(fig)  # Close the figure to avoid displaying it now
    
    print("...Plotting complete. Plots saved as PNG files.")
    # Combine all data
    combined_df = pd.concat(all_scenario_data, ignore_index=True)
    # Save combined raw data
    output_filename_raw = 'sim_data_raw.csv'
    combined_df.to_csv(output_filename_raw, index=False)
    print(f"\nCombined raw data saved to {output_filename_raw}")
    # --- Display Stats ---
    print("\n--- Data Summary ---")
    print(f"Total data points: {len(combined_df)}")
    print("\nScenario Distribution:")
    print(combined_df['Scenario'].value_counts())
    print("\nEngine Fault State Distribution:")
    print(combined_df['Engine_Fault_State'].value_counts())
    print("\nEGT Sensor Health Distribution:")
    print(combined_df['EGT_Sensor_Health'].value_counts())
    print("\nVibration Sensor Health Distribution:")
    print(combined_df['Vibration_Sensor_Health'].value_counts())
    print("\nNaN Counts per Sensor Column:")
    print(combined_df.isnull().sum())
    print("\nSample Data:")
    print(combined_df.sample(5))
    print("\n--- Simulation script finished ---")