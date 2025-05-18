import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- CONFIGURATION ---
STATE_ORDER = ["Healthy", "Degrading", "Critical"]
CRIT_THRESH = 0.76
WARN_THRESH = 0.24
HEALTH_NODE = "Engine_Core_Health"  # Must match your data

# --- Load your DataFrame of marginal probabilities ---
# Assumes CSV was saved like: all_probs.to_csv("all_probs_macroF1.csv")
df = pd.read_csv("all_probs_macroF1.csv")  # <-- Replace with your actual path

# --- Ground truth labels (must be available in original data) ---
# If labels are not in this file, load them separately
true_labels = df[HEALTH_NODE].tolist()

# --- Extract probability columns ---
p_crit = df[f"P({HEALTH_NODE}=Critical)"]
p_degr = df[f"P({HEALTH_NODE}=Degrading)"]

# --- Label mapping logic ---
def label_from_thresh(row):
    if row[f"P({HEALTH_NODE}=Critical)"] > CRIT_THRESH:
        return "Critical"
    elif row[f"P({HEALTH_NODE}=Degrading)"] > WARN_THRESH:
        return "Degrading"
    else:
        return "Healthy"

pred_labels = df.apply(label_from_thresh, axis=1)

# --- Confusion Matrix ---
cm = confusion_matrix(true_labels, pred_labels, labels=STATE_ORDER)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STATE_ORDER)

plt.figure(figsize=(6, 5))
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix (Tuned for Macro-F1)")
plt.tight_layout()
plt.savefig("confusion_macroF1.png", dpi=300)
plt.show()
