# config.py

# Simulation Timing
SIMULATION_DURATION_MINUTES = 30
DATA_FREQUENCY_HZ = 1
FAULT_ONSET_FRACTION = 0.4
FAULT_PROGRESSION_FRACTION = 0.3  # How long fault takes to fully appear
SENSOR_FAILURE_ONSET_FRACTION = 0.6
SENSOR_FAILURE_PROGRESSION_FRACTION = 0.2

# General Noise & Dropout
DEFAULT_NOISE_STD_FACTOR = 0.01  # Standard deviation as a fraction of the target value
DROPOUT_RATE = 0.02  # 2% dropout chance per sensor reading

# --- Sensor Specific Parameters ---
PARAMS = {
    'EGT': {
        'unit': 'C',
        'cruise_target': 900.0,
        'low_thresh': 850.0,
        'high_thresh': 950.0,
        'min_val': 400.0,
        'max_val': 1100.0,
        'noise_std': 5.0,  # Absolute noise level factor
        'ramp_factor': 0.02,  # Thermal inertia
        'bearing_wear_increase': 50.0,  # How much it increases in BearingWear
        'fail_drift': 100.0,  # How much sensor drifts when failing
        'fail_noise_factor': 5.0,  # Multiplier for noise when failing
        'has_hmm': True,
    },
    'N2': {
        'unit': '%RPM',
        'cruise_target': 90.0,
        'low_thresh': 85.0,
        'high_thresh': 95.0,
        'min_val': 50.0,
        'max_val': 110.0,
        'noise_std_factor': 0.005,  # N2 is usually stable
        'bearing_wear_decrease': 0.5,  # Subtle effect
        'has_hmm': False,
    },
    'OilPressure': {
        'unit': 'PSI',
        'cruise_target': 55.0,
        'low_thresh': 40.0,
        'high_thresh': 70.0,  # High not usually a fault symptom
        'min_val': 0.0,
        'max_val': 90.0,
        'noise_std_factor': 0.02,
        'oil_leak_target': 15.0,  # Pressure drops to this during leak
        'ramp_factor': 0.05,  # Pressure change isn't instant
        'has_hmm': False,
    },
    'Vibration': {  # Parameters common to Vib1 & Vib2
        'unit': 'IPS',
        'cruise_target': 1.0,
        'low_thresh': 0.5,
        'high_thresh': 1.5,
        'min_val': 0.0,
        'max_val': 5.0,
        'noise_std': 0.1,  # Absolute noise
        'bearing_wear_increase': 1.0,  # How much it increases in BearingWear
        'ramp_factor': 0.1,  # Mechanical changes not intant
        'fail_drift': 0.8,
        'fail_noise_factor': 4.0,
        'mrf_correlation_noise_std': 0.05,  # Additional noise specific to each sensor for MRF
        'has_hmm': True,
    }
}