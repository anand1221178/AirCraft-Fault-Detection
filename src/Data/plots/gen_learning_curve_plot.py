# NOT USED#
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from matplotlib.ticker import MaxNLocator

STATE_ORDER = ["Healthy", "Degrading", "Critical"]

def load_fold_predictions(prefix="DBN_learned_fold", n_folds=10):
    fold_data = []
    for i in range(n_folds):
        fname = f"{prefix}{i}.csv"
        if not os.path.exists(fname):
            print(f"[warn] Missing file: {fname}")
            continue
        df = pd.read_csv(fname)
        y_true = df["true"]
        y_pred = df[[f"P(Engine_Core_Health={s})" for s in STATE_ORDER]].idxmax(axis=1)
        y_pred = y_pred.str.extract(r"=(.*)")[0]
        fold_data.append((y_true, y_pred))
    return fold_data

def plot_learning_curve(fold_data, save_path="learning_curve_f1breakdown.png"):
    macro_f1s = []
    per_class = {s: [] for s in STATE_ORDER}
    for k in range(1, len(fold_data)+1):
        y_true_all = pd.concat([fold_data[i][0] for i in range(k)])
        y_pred_all = pd.concat([fold_data[i][1] for i in range(k)])
        macro_f1s.append(f1_score(y_true_all, y_pred_all, average="macro"))
        scores = f1_score(y_true_all, y_pred_all, average=None, labels=STATE_ORDER)
        for i, s in enumerate(STATE_ORDER):
            per_class[s].append(scores[i])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(range(1, len(macro_f1s)+1), macro_f1s, label="Macro F1", lw=2, marker='o')
    ax1.set_xlabel("Folds used for training")
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0, 1.05)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    for s in STATE_ORDER:
        ax1.plot(range(1, len(macro_f1s)+1), per_class[s], label=f"F1: {s}", linestyle="--", marker='.')

    ax1.set_title("Learning Curve: Macro-F1 and Per-Class Breakdown")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    folds = load_fold_predictions()
    plot_learning_curve(folds)
    print("Saved plot to: learning_curve_f1breakdown.png")