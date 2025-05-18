# File: run_experiment.py
# Description: Main pipeline for running DBN, MRF, Vanilla, and Rule-Based models on C-MAPSS

import sys
import pandas as pd
import os 
import numpy as np 
import matplotlib.pyplot as plt 
import pgmpy 
from sklearn.metrics import accuracy_score


sys.path.append("./Data_Gen")
sys.path.append("./DBN")
sys.path.append("./PreProcessing")
sys.path.append("./Utils")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
CMAPS_DATA_DIR = os.path.join(SCRIPT_DIR, "Data", "C-MAPSS") 

print(f"[run_experiment.py] CORRECTED CMAPS_DATA_DIR: {CMAPS_DATA_DIR}") # Added this for confirmation


from sim_data import prepare_cmaps_data
from mrf_model import temporal_mrf_smoothing
from dbn_inference import infer_marginals_dataframe
from evaluation import evaluate_predictions
from config import OBSERVATION_NODES, HEALTH_NODE, N_BINS, STATE_ORDER
from sklearn.metrics import accuracy_score, f1_score

from dbn_model import create_cmaps_dbn, learn_cpts_from_data


print("pgmpy:", pgmpy.__version__, sys.version)

    
def run_learned_dbn_pipeline() -> None:
    print("\n▶  Learned-CPD DBN pipeline (MLE, no hot-fix)")

    # 0 ▸  load + optional smoothing
    train_df, _, _ = prepare_cmaps_data(base_dir=CMAPS_DATA_DIR)
    for obs in OBSERVATION_NODES:
        if "vibration" in obs.lower():
            raw = obs.replace("_disc", "")
            if raw in train_df.columns:
                train_df[raw] = temporal_mrf_smoothing(train_df[raw])

    # 1 ▸  learn CPDs
    dbn = learn_cpts_from_data(create_cmaps_dbn(), train_df)

    # 2 ▸  per-unit inference
    prob_frames, true_labels = [], []
    for unit in train_df["unit"].unique():
        df_u = train_df[train_df["unit"] == unit]
        if df_u.empty:
            continue
        prob_frames.append(infer_marginals_dataframe(df_u, dbn))
        true_labels.extend(df_u[HEALTH_NODE].tolist())

    all_probs = pd.concat(prob_frames, ignore_index=True)
    smooth = all_probs.rolling(window=3, center=True, min_periods=1).mean()

    # 3 ▸  helper
    crit_col = f"P({HEALTH_NODE}={STATE_ORDER[2]})"
    deg_col  = f"P({HEALTH_NODE}={STATE_ORDER[1]})"

    def label_from_thresh(row, fail_thr, warn_thr):
        if row[crit_col] > fail_thr:
            return STATE_ORDER[2]            # Critical
        if row[deg_col]  > warn_thr:
            return STATE_ORDER[1]            # Degrading
        return STATE_ORDER[0]                # Healthy

    # 4 ▸  fixed 0.70 / 0.50 baseline
    preds_fixed = [label_from_thresh(r, 0.70, 0.50) for _, r in smooth.iterrows()]
    print("\n--- Classification report (fixed thresholds 0.70 / 0.50) ---")
    evaluate_predictions(true_labels, preds_fixed, labels=STATE_ORDER,
                         output_prefix="DBN_learned_fixed")

    # 5 ▸  accuracy-optimised grid (original behaviour)
    best_acc, best_f, best_w = 0, None, None
    for f_thr in np.linspace(0.20, 0.95, 10):
        for w_thr in np.linspace(0.20, 0.95, 10):
            preds = [label_from_thresh(r, f_thr, w_thr) for _, r in smooth.iterrows()]
            acc   = accuracy_score(true_labels, preds)
            if acc > best_acc:
                best_acc, best_f, best_w = acc, f_thr, w_thr
    print(f"\nBest tuned **accuracy** = {best_acc:.4f}  "
          f"(Fail>{best_f:.2f}, Warn>{best_w:.2f})")
    preds_acc = [label_from_thresh(r, best_f, best_w) for _, r in smooth.iterrows()]
    evaluate_predictions(true_labels, preds_acc, labels=STATE_ORDER,
                         output_prefix="DBN_learned_accTuned")

    # 6 ▸  macro-F1 grid (finer)
    best_f1, best_f, best_w = 0, None, None
    for f_thr in np.linspace(0.20, 0.90, 36):             # 0.02 step
        for w_thr in np.linspace(0.05, f_thr - 0.05, 18): # guarantee warn < fail
            preds = [label_from_thresh(r, f_thr, w_thr) for _, r in all_probs.iterrows()]
            f1    = f1_score(true_labels, preds, average="macro")
            if f1 > best_f1:
                best_f1, best_f, best_w = f1, f_thr, w_thr
    print(f"\nBest tuned **macro-F1** = {best_f1:.4f}  "
          f"(Fail>{best_f:.2f}, Warn>{best_w:.2f})")
    preds_f1 = [label_from_thresh(r, best_f, best_w) for _, r in all_probs.iterrows()]
    evaluate_predictions(true_labels, preds_f1, labels=STATE_ORDER,
                         output_prefix="DBN_learned_macroF1")
        # 7 ▸ save for macro-F1 confusion matrix plot
    smooth[HEALTH_NODE] = true_labels
    smooth.to_csv("Data/plots/all_probs_macroF1.csv", index=False)

    # 8 ▸ save Unit 1 timeline posteriors
    unit1_df = train_df[train_df["unit"] == 1]
    unit1_post = infer_marginals_dataframe(unit1_df, dbn)
    unit1_post["cycle"] = unit1_df.index
    unit1_post["true"] = unit1_df[HEALTH_NODE].values
    unit1_post.to_csv("Data/plots/unit1_posteriors.csv", index=False)
    # 9 ▸ Save full inference output (unsmoothed) for threshold heatmap
    all_probs["cycle"] = np.arange(len(all_probs))
    all_probs["true"] = true_labels
    all_probs.to_csv("Data/plots/all_probs_full.csv", index=False)





from sklearn.model_selection import StratifiedKFold

def save_fold_predictions() -> None:
    print("\n▶  Generating 10-fold posterior predictions for learning curve")

    train_df, _, _ = prepare_cmaps_data(base_dir=CMAPS_DATA_DIR)

    # Smooth vibration needed
    for obs in OBSERVATION_NODES:
        if "vibration" in obs.lower():
            raw = obs.replace("_disc", "")
            if raw in train_df.columns:
                train_df[raw] = temporal_mrf_smoothing(train_df[raw])

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y = train_df[HEALTH_NODE].tolist()
    folds = list(skf.split(train_df, y))

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {fold_idx}...")

        df_train = train_df.iloc[train_idx].copy()
        df_test  = train_df.iloc[test_idx].copy()

        dbn = learn_cpts_from_data(create_cmaps_dbn(), df_train)

        rows = []
        for unit in df_test["unit"].unique():
            df_u = df_test[df_test["unit"] == unit]
            if df_u.empty:
                continue
            probs = infer_marginals_dataframe(df_u, dbn)
            
            for t, (_, row) in enumerate(probs.iterrows()):
                post = row.values.tolist()
                true = df_u.iloc[t][HEALTH_NODE]
                rows.append(post + [t, true])

        df_out = pd.DataFrame(rows, columns=[
            f"P({HEALTH_NODE}={s})" for s in STATE_ORDER
        ] + ["cycle", "true"])
        df_out.to_csv(f"DBN_learned_fold{fold_idx}.csv", index=False)

    print("✅ All 10 folds saved as DBN_learned_fold0.csv → fold9.csv")

def label_from_thresh(row, tau_c=0.76, tau_d=0.24):
    if row[f"P({HEALTH_NODE}=Critical)"] > tau_c:
        return "Critical"
    elif row[f"P({HEALTH_NODE}=Degrading)"] > tau_d:
        return "Degrading"
    return "Healthy"



if __name__ == "__main__":
    run_learned_dbn_pipeline()
    save_fold_predictions()
