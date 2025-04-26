# evaluation.py

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define consistent labels (adjust if your prediction strings differ slightly)
# Using ground truth labels for calculation, map predictions if needed
LABELS_ENGINE_FAULT = ['Normal', 'OilLeak_Active', 'BearingWear_Active'] 
# Note: We evaluate sensor failure detection separately or map predictions like 'EGTSensorFail_Predicted' to 'Normal' for this engine fault evaluation.

LABELS_SENSOR_HEALTH = ['OK', 'Degraded', 'Failed']

def map_predictions_to_gt_engine(y_pred_series):
    """ Maps detailed prediction strings to basic GT engine fault labels. """
    mapping = {
        'Normal_Predicted': 'Normal',
        'OilLeak_Predicted': 'OilLeak_Active',
        'BearingWear_Predicted': 'BearingWear_Active',
        'EGTSensorFail_Predicted': 'Normal', # Sensor failure means engine was Normal
        'VibSensorFail_Predicted': 'Normal', # Sensor failure means engine was Normal
        'Error': 'Unknown' # Or handle errors differently
    }
    return y_pred_series.map(mapping).fillna('Unknown')

def map_predictions_to_gt_sensor(y_pred_series, sensor_type):
    """ Maps prediction strings relevant to sensor health back to GT labels. """
    if sensor_type == 'EGT':
        fail_pred = 'EGTSensorFail_Predicted'
    elif sensor_type == 'Vibration':
        fail_pred = 'VibSensorFail_Predicted'
    else:
        return pd.Series(['Unknown'] * len(y_pred_series), index=y_pred_series.index) # Should not happen

    # Simple mapping: If sensor failure predicted -> Failed, otherwise -> OK
    # Note: Doesn't explicitly map to 'Degraded' from predictions
    mapping = {
        fail_pred: 'Failed',
        # All other predictions imply the sensor itself was okay *from the prediction perspective*
        'Normal_Predicted': 'OK',
        'OilLeak_Predicted': 'OK',
        'BearingWear_Predicted': 'OK',
        'Error': 'Unknown'
    }
     # Add the opposite sensor failure type mapping to OK as well
    if sensor_type == 'EGT': mapping['VibSensorFail_Predicted'] = 'OK'
    if sensor_type == 'Vibration': mapping['EGTSensorFail_Predicted'] = 'OK'

    return y_pred_series.map(mapping).fillna('Unknown')


def calculate_metrics(y_true, y_pred, labels):
    """ Calculates accuracy, precision, recall, f1 (weighted). """
    # Remove 'Unknown' or other unhandled labels before calculation if necessary
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    if not valid_indices.any():
        print("Warning: No valid matching labels found between true and predicted values.")
        return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]

    if len(y_true_filtered) == 0:
         print("Warning: No overlapping valid labels after filtering.")
         return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=labels, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def generate_classification_report(y_true, y_pred, labels, target_names=None):
    """ Generates and returns the classification report string. """
    # Filter labels similar to calculate_metrics
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    
    if len(y_true_filtered) == 0:
         return "No valid matching labels found for classification report."

    if target_names is None:
        target_names = labels
        
    # Ensure labels passed to report are only those present in the filtered data
    present_labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
    present_target_names = [name for name, label in zip(target_names, labels) if label in present_labels]


    report = classification_report(
        y_true_filtered, y_pred_filtered, labels=present_labels, target_names=present_target_names, zero_division=0
    )
    return report

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', save_path=None):
    """ Generates and optionally saves a confusion matrix plot. """
     # Filter labels
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]

    if len(y_true_filtered) == 0:
         print("Warning: No valid matching labels found for confusion matrix.")
         return

    # Calculate matrix ensuring labels order
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    else:
        plt.show()


# Example usage (if run standalone for testing)
if __name__ == "__main__":
    # Dummy data
    y_true_engine = pd.Series(['Normal', 'Normal', 'OilLeak_Active', 'OilLeak_Active', 'BearingWear_Active', 'Normal', 'Normal'])
    y_pred_engine_raw = pd.Series(['Normal_Predicted', 'OilLeak_Predicted', 'OilLeak_Predicted', 'BearingWear_Predicted', 'BearingWear_Predicted', 'EGTSensorFail_Predicted', 'Error'])
    
    y_true_egt_health = pd.Series(['OK', 'OK', 'OK', 'OK', 'OK', 'Failed', 'Failed'])
    y_pred_health_raw = pd.Series(['Normal_Predicted', 'OilLeak_Predicted', 'BearingWear_Predicted', 'VibSensorFail_Predicted', 'Normal_Predicted', 'EGTSensorFail_Predicted', 'Error'])


    # Map predictions
    y_pred_engine_mapped = map_predictions_to_gt_engine(y_pred_engine_raw)
    y_pred_egt_health_mapped = map_predictions_to_gt_sensor(y_pred_health_raw, 'EGT')

    print("--- Engine Fault Evaluation ---")
    print("True:", y_true_engine.tolist())
    print("Pred:", y_pred_engine_mapped.tolist())
    engine_metrics = calculate_metrics(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT)
    print("Metrics:", engine_metrics)
    print("Classification Report:\n", generate_classification_report(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT))
    plot_confusion_matrix(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT, title='Engine Fault CM (Example)')

    print("\n--- EGT Sensor Health Evaluation ---")
    print("True:", y_true_egt_health.tolist())
    print("Pred:", y_pred_egt_health_mapped.tolist())
    egt_health_metrics = calculate_metrics(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH)
    print("Metrics:", egt_health_metrics)
    print("Classification Report:\n", generate_classification_report(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH))
    plot_confusion_matrix(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH, title='EGT Sensor Health CM (Example)')