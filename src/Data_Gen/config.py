# File: config.py
# Description: Configuration for C-MAPSS dataset modeling

# ===== Dataset Settings =====
DATASET_ID = "FD001"  # Options: FD001, FD002, FD003, FD004

# ===== Health Label Thresholds =====
RUL_THRESHOLDS = {
    "Healthy": 120,
    "Degrading": 60,
    "Critical": 0,
}

STATE_ORDER = ['Healthy', 'Degrading', 'Critical']   # global truth
STATE_INDEX = {s:i for i, s in enumerate(STATE_ORDER)}


# ===== Top Sensors to Use =====
# These were found to be useful in prior C-MAPSS studies (can be changed)
SELECTED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_11",
    "sensor_15",
    "sensor_17",
    "sensor_20",
]

# ===== Discretization Settings =====
# config.py
BINNING_SCHEME = {
    "sensor_2":  (4, "quantile"),
    "sensor_11": (4, "quantile"),
    "sensor_17": (4, "quantile"),
    "sensor_20": (4, "quantile"),
    "sensor_3":  (4, "kmeans"),
    "sensor_4":  (4, "kmeans"),
    "sensor_7":  (4, "kmeans"),
    "sensor_8":  (2, "uniform"),
    "sensor_15": (2, "uniform"),
}
DEFAULT_N_BINS       = 3
DEFAULT_BIN_STRATEGY = "quantile"


# Optional: Manual bin edges (if you prefer fixed intervals)
# Format: {'sensor_x': [min, b1, b2, ..., max]}
MANUAL_BINS = {}

# ===== Node Naming for DBN =====
HEALTH_NODE = "Engine_Core_Health" # <<<< CHANGE THIS
OBSERVATION_NODES = [f"{sensor}_disc" for sensor in SELECTED_SENSORS]

# ------------------------------------------------------------------
# Back-compat aliases for older modules
# ------------------------------------------------------------------
N_BINS           = DEFAULT_N_BINS
BINNING_STRATEGY = DEFAULT_BIN_STRATEGY

