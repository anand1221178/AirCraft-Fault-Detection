import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load prediction file with true labels and predicted labels
df = pd.read_csv("all_probs_macroF1.csv")


def label_from_thresh(row, tau_c=0.76, tau_d=0.24):
    if row["P(Engine_Core_Health=Critical)"] > tau_c:
        return "Critical"
    elif row["P(Engine_Core_Health=Degrading)"] > tau_d:
        return "Degrading"
    return "Healthy"

df["pred"] = df.apply(label_from_thresh, axis=1)

# Confusion matrix
labels = ["Healthy", "Degrading", "Critical"]
cm = confusion_matrix(df["Engine_Core_Health"], df["pred"], labels=labels)

# FN and FP for each class
fn = cm.sum(axis=1) - cm.diagonal()  # False Negatives per class
fp = cm.sum(axis=0) - cm.diagonal()  # False Positives per class

# Plot
x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 4.5))
plt.bar(x, fp, width, label='False Positives', color='orange')
plt.bar([i + width for i in x], fn, width, label='False Negatives', color='steelblue')

plt.xticks([i + width / 2 for i in x], labels)
plt.ylabel("Count")
plt.title("Per-Class Error Breakdown")
plt.legend()
plt.tight_layout()
plt.savefig("error_breakdown_per_class.png", dpi=300)
plt.close()
