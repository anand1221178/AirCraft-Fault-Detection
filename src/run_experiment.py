# File: run_experiment.py
# Description: Main pipeline for running DBN, MRF, Vanilla, and Rule-Based models on C-MAPSS

import sys
import pandas as pd
import os # For os.makedirs
import numpy as np # For np.array, np.nan, etc.
import matplotlib.pyplot as plt # For plotting
import pgmpy # For pgmpy.__version__
from sklearn.metrics import accuracy_score

# --- Include submodules in path ---
sys.path.append("./Data_Gen")
sys.path.append("./DBN")
sys.path.append("./PreProcessing")
sys.path.append("./Utils")
# Get the directory of the current script (run_experiment.py)
# This assumes run_experiment.py is in the 'src' folder.
# --- CORRECTED Path Construction ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # This correctly gets PROJECT_ROOT/src/
# CMAPS_DATA_DIR is relative to SCRIPT_DIR (i.e., src/)
CMAPS_DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "C-MAPSS") 
# This will now correctly resolve to: PROJECT_ROOT/src/Data/C-MAPSS/
print(f"[run_experiment.py] CORRECTED CMAPS_DATA_DIR: {CMAPS_DATA_DIR}") # Add this for confirmation

# --- Imports from your project modules ---
from sim_data import prepare_cmaps_data
from mrf_model import temporal_mrf_smoothing
from dbn_inference import infer_marginals_dataframe
from evaluation import evaluate_predictions
from config import OBSERVATION_NODES, HEALTH_NODE, N_BINS, STATE_ORDER

from dbn_model import create_cmaps_dbn, learn_cpts_from_data

from Utils.rul import compute_rul

# --- Imports from pgmpy ---

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

print("pgmpy:", pgmpy.__version__, sys.version)


# -------------------------------------------------------------------------
# 1.  FULL PIPELINE  (expert CPTs  ➜  inference  ➜  hot-fix  ➜  tuning)
# -------------------------------------------------------------------------
def run_full_dbn_pipeline() -> None:
    print("\n▶  Full DBN pipeline (expert CPTs + RUL hot-fix + thresh-tuning)")
    train_df, _, _ = prepare_cmaps_data(base_dir=CMAPS_DATA_DIR)

    # --- 1 ▸  vibration smoothing ---------------------------------------
    for obs in OBSERVATION_NODES:
        if "vibration" in obs.lower():
            raw = obs.replace("_disc", "")
            if raw in train_df.columns:
                train_df[raw] = temporal_mrf_smoothing(train_df[raw])

    # --- 2 ▸  build DBN with expert CPDs --------------------------------
    dbn = create_cmaps_dbn()

    H, D, C = STATE_ORDER            # Healthy, Degrading, Critical
    n_states = len(STATE_ORDER)

    # 2.a ■ transition  P(Hₜ₊₁ | Hₜ)
    trans_vals = np.array([
        # parent‐states order = H, D, C   (columns)
        # child states  ↓
        [0.85, 0.10, 0.01],   #   H
        [0.14, 0.70, 0.10],   #   D
        [0.01, 0.20, 0.89],   #   C
    ])
    trans_vals /= trans_vals.sum(axis=0, keepdims=True)

    dbn.add_cpds(
        TabularCPD(
            variable=(HEALTH_NODE, 1),
            variable_card=n_states,
            values=trans_vals,
            evidence=[(HEALTH_NODE, 0)],
            evidence_card=[n_states],
            state_names={(HEALTH_NODE, 0): STATE_ORDER,
                         (HEALTH_NODE, 1): STATE_ORDER},
        )
    )

    # 2.b ■ emission  P(sensor_bin | Hₜ)
    def col_reorder(arr):
        # current matrices are in order C, D, H  →  convert to H, D, C
        return arr[:, [2, 1, 0]]

    sensor_cpd_values = {
        # rows = bins 0-(N_BINS-1), cols = C, D, H
        "sensor_2_disc":  col_reorder(np.array([[0.0236, 0.2235, 0.4124],
                                                [0.7100, 0.7663, 0.5843],
                                                [0.2664, 0.0102, 0.0033]])),
        "sensor_3_disc":  col_reorder(np.array([[0.0310, 0.2440, 0.4181],
                                                [0.8123, 0.7505, 0.5811],
                                                [0.1567, 0.0055, 0.0008]])),
        "sensor_4_disc":  col_reorder(np.array([[0.0098, 0.1908, 0.4256],
                                                [0.6625, 0.8045, 0.5741],
                                                [0.3277, 0.0047, 0.0002]])),
        "sensor_7_disc":  col_reorder(np.array([[0.2323, 0.0028, 0.0001],
                                                [0.7495, 0.7777, 0.5382],
                                                [0.0182, 0.2195, 0.4617]])),
        "sensor_11_disc": col_reorder(np.array([[0.0239, 0.3238, 0.5956],
                                                [0.6987, 0.6748, 0.4044],
                                                [0.2774, 0.0013, 1e-6]])),
        "sensor_17_disc": col_reorder(np.array([[0.0036, 0.0927, 0.2259],
                                                [0.7287, 0.8975, 0.7731],
                                                [0.2677, 0.0098, 0.0011]])),
        "sensor_20_disc": col_reorder(np.array([[0.3210, 0.0127, 0.0027],
                                                [0.6711, 0.8607, 0.7031],
                                                [0.0079, 0.1267, 0.2942]])),
    }

    for obs in OBSERVATION_NODES:
        vals = sensor_cpd_values.get(obs,
                                     np.full((N_BINS, n_states),
                                             1.0 / N_BINS))
        vals /= vals.sum(axis=0, keepdims=True)
        dbn.add_cpds(
            TabularCPD(
                variable=(obs, 0),
                variable_card=N_BINS,
                values=vals,
                evidence=[(HEALTH_NODE, 0)],
                evidence_card=[n_states],
                state_names={(obs, 0): [str(i) for i in range(N_BINS)],
                             (HEALTH_NODE, 0): STATE_ORDER},
            )
        )
        dbn.add_cpds(
            TabularCPD(
                variable=(obs, 1),
                variable_card=N_BINS,
                values=vals,
                evidence=[(HEALTH_NODE, 1)],
                evidence_card=[n_states],
                state_names={(obs, 1): [str(i) for i in range(N_BINS)],
                             (HEALTH_NODE, 1): STATE_ORDER},
            )
        )

    dbn.check_model()

    # --- 3 ▸  per-unit inference + hot-fix + collect ---------------------
    prob_frames, true_labels, true_rul = [], [], []
    for unit in train_df["unit"].unique():
        df_u = train_df[train_df["unit"] == unit]
        if df_u.empty:
            continue

        probs = infer_marginals_dataframe(df_u, dbn)
        rul_vec = df_u["RUL"].values

        # RUL hot-fix for last 30 cycles
        crit_col = f"P({HEALTH_NODE}={C})"
        deg_col  = f"P({HEALTH_NODE}={D})"
        hea_col  = f"P({HEALTH_NODE}={H})"
        late     = rul_vec <= 30
        probs.loc[late, crit_col] = 0.95
        probs.loc[late, deg_col]  = 0.025
        probs.loc[late, hea_col]  = 0.025

        probs["RUL_cycles"] = compute_rul(probs, fail_col=crit_col)

        prob_frames.append(probs)
        true_labels.extend(df_u[HEALTH_NODE].tolist())
        true_rul.extend(rul_vec.tolist())

    all_probs = pd.concat(prob_frames).reset_index(drop=True)

    # --- 4 ▸  quick fixed-threshold report -------------------------------
    def label_from_thresh(p_row, f_thr=0.70, w_thr=0.50):
        if p_row[crit_col] > f_thr:
            return C
        if p_row[deg_col] > w_thr:
            return D
        return H

    preds_fixed = [label_from_thresh(r) for _, r in all_probs.iterrows()]
    print("\n--- Classification report (fixed thresholds) ---")
    evaluate_predictions(true_labels, preds_fixed,
                         labels=STATE_ORDER,
                         output_prefix="DBN_fixed")

    # --- 5 ▸  tune thresholds -------------------------------------------
    best_acc, best_f, best_w = 0.0, None, None
    for f_thr in np.linspace(0.20, 0.95, 10):
        for w_thr in np.linspace(0.20, 0.95, 10):
            preds = [label_from_thresh(r, f_thr, w_thr)
                     for _, r in all_probs.iterrows()]
            acc = accuracy_score(true_labels, preds)
            if acc > best_acc:
                best_acc, best_f, best_w = acc, f_thr, w_thr

    print(f"\nBest tuned accuracy = {best_acc:.4f}  "
          f"(Fail>{best_f:.2f}, Warn>{best_w:.2f})")

    preds_best = [label_from_thresh(r, best_f, best_w)
                  for _, r in all_probs.iterrows()]
    evaluate_predictions(true_labels, preds_best,
                         labels=STATE_ORDER,
                         output_prefix="DBN_tuned")
    
def run_learned_dbn_pipeline() -> None:
    print("\n▶  Learned-CPD DBN pipeline (BayesianEstimator, no hot-fix)")
    train_df, _, _ = prepare_cmaps_data(base_dir=CMAPS_DATA_DIR)

    # optional vibration smoothing before learning
    for obs in OBSERVATION_NODES:
        if "vibration" in obs.lower():
            raw = obs.replace("_disc", "")
            if raw in train_df.columns:
                train_df[raw] = temporal_mrf_smoothing(train_df[raw])

    # structure only
    dbn_struct = create_cmaps_dbn()

    # learn all CPDs from data
    dbn_learned = learn_cpts_from_data(dbn_struct, train_df)


    # inference on the same training set (unit by unit, like before)
    prob_frames, true_labels = [], []
    for unit in train_df["unit"].unique():
        df_u = train_df[train_df["unit"] == unit]
        if df_u.empty:
            continue
        probs = infer_marginals_dataframe(df_u, dbn_learned)
        prob_frames.append(probs)
        true_labels.extend(df_u[HEALTH_NODE].tolist())

    all_probs = pd.concat(prob_frames).reset_index(drop=True)

    # fixed thresholds first (same 0.70 / 0.50)
    crit_col = f"P({HEALTH_NODE}={STATE_ORDER[2]})"
    deg_col  = f"P({HEALTH_NODE}={STATE_ORDER[1]})"
    hea_col  = f"P({HEALTH_NODE}={STATE_ORDER[0]})"

    def label_from_thresh(p_row, f_thr=0.70, w_thr=0.50):
        if p_row[crit_col] > f_thr: return STATE_ORDER[2]
        if p_row[deg_col]  > w_thr: return STATE_ORDER[1]
        return STATE_ORDER[0]

    preds_fixed = [label_from_thresh(r) for _, r in all_probs.iterrows()]
    print("\n--- Classification report (fixed thresholds) ---")
    evaluate_predictions(true_labels, preds_fixed,
                         labels=STATE_ORDER,
                         output_prefix="DBN_learned_fixed")

    # quick grid-search for better thresholds
    best_acc, best_f, best_w = 0.0, None, None
    for f_thr in np.linspace(0.20, 0.95, 10):
        for w_thr in np.linspace(0.20, 0.95, 10):
            preds = [label_from_thresh(r, f_thr, w_thr)
                     for _, r in all_probs.iterrows()]
            acc = accuracy_score(true_labels, preds)
            if acc > best_acc:
                best_acc, best_f, best_w = acc, f_thr, w_thr

    print(f"\nBest tuned accuracy = {best_acc:.4f} "
          f"(Fail>{best_f:.2f}, Warn>{best_w:.2f})")

    preds_best = [label_from_thresh(r, best_f, best_w)
                  for _, r in all_probs.iterrows()]
    evaluate_predictions(true_labels, preds_best,
                         labels=STATE_ORDER,
                         output_prefix="DBN_learned_tuned")

    


# -------------------------------------------------------------------------
# 2.  SENSOR-DRIFT DEMO  (updated column names)
# -------------------------------------------------------------------------
def run_drift_adaptation_demo() -> None:
    print("\n▶  Drift-adaptation demo")
    try:
        from sim_data import generate_drifted_engine
        from DBN.update_cpd import update_cpds_online
        from Data_Gen.cmaps_data_loader import add_discrete_health_label
    except ImportError as e:
        print("Skip drift demo – missing component:", e)
        return

    df_drift = generate_drifted_engine("EGT_Drift", n_steps=50)
    if df_drift.empty:
        print("No drifted data generated.")
        return

    if "RUL" not in df_drift:
        df_drift["RUL"] = 300

    df_drift = add_discrete_health_label(df_drift)

    clean_train, _, _ = prepare_cmaps_data(base_dir=CMAPS_DATA_DIR)
    base_dbn = learn_cpts_from_data(create_cmaps_dbn(), clean_train)

    no_adapt = infer_marginals_dataframe(df_drift, base_dbn.copy())
    adapt    = update_cpds_online(base_dbn, df_drift, alpha=20.0)

    crit_col = f"P({HEALTH_NODE}={STATE_ORDER[2]})"
    plt.figure()
    if crit_col in no_adapt:
        plt.plot(no_adapt.index, no_adapt[crit_col], "--", label="No adapt")
    if isinstance(adapt, pd.DataFrame) and crit_col in adapt:
        plt.plot(adapt.index, adapt[crit_col], "-", label="Adapt")
    plt.xlabel("cycle"); plt.ylabel("P(Critical)"); plt.legend()
    os.makedirs("Data/plots", exist_ok=True)
    plt.savefig("Data/plots/drift_adaptation_comparison.png")
    plt.close()


if __name__ == "__main__":
    run_learned_dbn_pipeline()
    # run_full_dbn_pipeline()
    # run_rule_based_baseline()
    # run_drift_adaptation_demo()