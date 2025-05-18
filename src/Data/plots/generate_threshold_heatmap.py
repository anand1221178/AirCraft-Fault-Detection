import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
import os

# Load all folds
dfs = [pd.read_csv(f"DBN_learned_fold{i}.csv") for i in range(10)]
df = pd.concat(dfs, ignore_index=True)
true = df["true"].astype(str)

pH = df["P(Engine_Core_Health=Healthy)"].values
pD = df["P(Engine_Core_Health=Degrading)"].values
pC = df["P(Engine_Core_Health=Critical)"].values

fail_range = np.linspace(0.20, 0.95, 16)
warn_range = np.linspace(0.05, 0.85, 17)

scores = np.full((len(warn_range), len(fail_range)), np.nan)

for i, warn in enumerate(warn_range):
    for j, fail in enumerate(fail_range):
        if warn >= fail:
            continue
        preds = []
        for pc, pd in zip(pC, pD):
            if pc > fail:
                preds.append("Critical")
            elif pd > warn:
                preds.append("Degrading")
            else:
                preds.append("Healthy")
        scores[i, j] = f1_score(true, preds, average="macro")

plt.figure(figsize=(10, 6))
sns.heatmap(scores, xticklabels=[f"{x:.2f}" for x in fail_range],
            yticklabels=[f"{y:.2f}" for y in warn_range],
            cmap="viridis", annot=True, fmt=".2f",
            cbar_kws={"label": "Macro-F1 Score"})
plt.xlabel("Failure Threshold (τ_C)")
plt.ylabel("Degrading Threshold (τ_D)")
plt.title("Macro-F1 Score Heatmap")
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/threshold_heatmap_f1.png", dpi=300)
plt.close()
