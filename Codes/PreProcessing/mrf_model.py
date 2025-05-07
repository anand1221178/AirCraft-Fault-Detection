# mrf_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def apply_simple_mrf_smoother(vib1_raw, vib2_raw, iterations=5, strength=0.5, preserve_mean=True):
    """
    Applies a simple iterative smoothing to Vib1 and Vib2 based on local consistency.
    Assumes Vib1 and Vib2 should generally be similar. Pulls differing values closer.

    Args:
        vib1_raw (pd.Series): Raw continuous Vib1 data (can contain NaNs).
        vib2_raw (pd.Series): Raw continuous Vib2 data (can contain NaNs).
        iterations (int): Number of smoothing iterations.
        strength (float): How strongly to pull values together (0 to 1). Higher is stronger.
        preserve_mean (bool): If True, tries to keep the mean of vib1+vib2 constant during smoothing.

    Returns:
        tuple[pd.Series, pd.Series]: Smoothed Vib1 and Vib2 series.
    """
    print(f"Applying MRF-like smoothing: iterations={iterations}, strength={strength}")
    
    # Handle NaNs: Forward fill then backward fill to estimate missing values for smoothing
    vib1_filled = vib1_raw.ffill().bfill()
    vib2_filled = vib2_raw.ffill().bfill()
    
    # Convert to numpy for faster iteration
    vib1_smooth = vib1_filled.to_numpy(copy=True)
    vib2_smooth = vib2_filled.to_numpy(copy=True)

    n_points = len(vib1_smooth)

    for _ in range(iterations):
        # Store previous state for comparison or simultaneous update
        vib1_prev_iter = vib1_smooth.copy()
        vib2_prev_iter = vib2_smooth.copy()

        for i in range(n_points):
            # Difference between the two sensors at this point
            diff = vib1_prev_iter[i] - vib2_prev_iter[i]
            
            # Calculate adjustment amount based on difference and strength
            adjustment = diff * strength / 2.0 # Divide by 2 as it applies to both sides

            # Apply adjustment
            new_vib1 = vib1_prev_iter[i] - adjustment
            new_vib2 = vib2_prev_iter[i] + adjustment

            #Preserve the local mean (prevents overall drift)
            if preserve_mean:
                 mean_adjustment = ((new_vib1 + new_vib2) - (vib1_prev_iter[i] + vib2_prev_iter[i])) / 2.0
                 new_vib1 -= mean_adjustment
                 new_vib2 -= mean_adjustment

            vib1_smooth[i] = new_vib1
            vib2_smooth[i] = new_vib2

    # Convert back to pandas Series, preserving original index
    vib1_final = pd.Series(vib1_smooth, index=vib1_raw.index, name='Vib1_IPS_Smoothed')
    vib2_final = pd.Series(vib2_smooth, index=vib2_raw.index, name='Vib2_IPS_Smoothed')


    # This ensures that actual sensor dropouts are still dropouts after smoothing.
    vib1_final[vib1_raw.isna()] = np.nan
    vib2_final[vib2_raw.isna()] = np.nan
    
    print("Smoothing complete.")
    return vib1_final, vib2_final

# --- Validation Part ---
if __name__ == "__main__":
    print("--- MRF Validation ---")
    # Setup
    DATA_DIR = '../Data' 
    RAW_DATA_FILE = os.path.join(DATA_DIR, 'sim_data_raw.csv')
    PLOT_SAVE_DIR = os.path.join(DATA_DIR, 'mrf_validation_plots')
    if not os.path.exists(PLOT_SAVE_DIR):
        os.makedirs(PLOT_SAVE_DIR)

    # Load raw data
    print(f"Loading raw data from {RAW_DATA_FILE}...")
    try:
        df_raw = pd.read_csv(RAW_DATA_FILE, parse_dates=['Timestamp'])
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_FILE}")
        exit()
    except Exception as e:
        print(f"Error loading raw data: {e}")
        exit()

    # Select a scenario with some noise/variation 
    scenario_to_plot = 'BearingWear' # Or 'Normal'
    plot_slice_df = df_raw[df_raw['Scenario'] == scenario_to_plot].head(200) # Plot first 200 steps

    if plot_slice_df.empty:
        print(f"Error: No data found for scenario '{scenario_to_plot}'")
        exit()

    print(f"Applying smoother to Vib1/Vib2 for scenario '{scenario_to_plot}'...")
    vib1_s, vib2_s = apply_simple_mrf_smoother(
        plot_slice_df['Vib1_IPS'], 
        plot_slice_df['Vib2_IPS'],
        iterations=10, # More iterations for potentially smoother result
        strength=0.6  # Adjust strength
    )

    # Plotting raw vs smoothed
    print("Plotting comparison...")
    plt.figure(figsize=(15, 8))
    
    plt.plot(plot_slice_df['Timestamp'], plot_slice_df['Vib1_IPS'], label='Vib1 Raw', color='blue', alpha=0.5, linestyle=':')
    plt.plot(plot_slice_df['Timestamp'], plot_slice_df['Vib2_IPS'], label='Vib2 Raw', color='orange', alpha=0.5, linestyle=':')
    
    plt.plot(plot_slice_df['Timestamp'], vib1_s, label='Vib1 Smoothed', color='blue', linewidth=1.5)
    plt.plot(plot_slice_df['Timestamp'], vib2_s, label='Vib2 Smoothed', color='orange', linewidth=1.5)

    plt.title(f'MRF Smoother Validation (Scenario: {scenario_to_plot})')
    plt.xlabel('Timestamp')
    plt.ylabel('Vibration (IPS)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    save_path = os.path.join(PLOT_SAVE_DIR, f'mrf_smoothing_{scenario_to_plot}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved MRF validation plot to {save_path}")