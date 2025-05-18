# File: evaluation.py
# Description: Evaluates prediction performance for C-MAPSS fault detection

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def evaluate_predictions(true_labels, predicted_labels, labels=None, output_prefix="output", save_dir="../Data/experiment_results/analysis_output"): # Add labels=None
    os.makedirs(save_dir, exist_ok=True)

    # Pass labels to classification_report
    report = classification_report(true_labels, predicted_labels, labels=labels, output_dict=False, zero_division=0) # Add zero_division
    print("Classification Report:\n", report)
    with open(os.path.join(save_dir, f"report_{output_prefix}.txt"), "w") as f:
        f.write(report)


    if labels is None: # If no specific order, derive from unique values
        labels = sorted(list(set(true_labels) | set(predicted_labels)))
    
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    cm_df = pd.DataFrame(cm, 
                           index=[f"True {l}" for l in labels],
                           columns=[f"Pred {l}" for l in labels])

    plt.figure(figsize=(max(6, len(labels)*2), max(5, len(labels)*1.8))) # Adjust size
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {output_prefix}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{output_prefix}.png"))
    plt.close()

# Example usage
if __name__ == "__main__":
    true = ["Healthy", "Degrading", "Critical", "Degrading", "Critical"]
    pred = ["Healthy", "Degrading", "Critical", "Healthy", "Critical"]
    evaluate_predictions(true, pred, output_prefix="test_eval")