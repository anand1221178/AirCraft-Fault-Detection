# evaluation.py

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
# Adjust path if evaluation.py is not in the 'Utils' directory relative to 'Data'
RESULTS_DIR = '../Data/experiment_results' 
RESULTS_FILE = os.path.join(RESULTS_DIR, 'experiment_results_raw.csv')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'analysis_output') # Subdir for evaluation outputs
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define consistent labels
LABELS_ENGINE_FAULT = ['Normal', 'OilLeak_Active', 'BearingWear_Active'] 
LABELS_SENSOR_HEALTH = ['OK', 'Degraded', 'Failed']

# Define models and their prediction column names
MODELS_TO_EVALUATE = {
    "Full_DBN": "Prediction_FullDBN",
    "Vanilla_DBN": "Prediction_VanillaDBN",
    "Rule_Based": "Prediction_RuleBased"
}
# Define which models produce sensor health predictions interpretable by our mapping
MODELS_WITH_SENSOR_PREDS = ["Full_DBN"] # Only the full DBN maps failure preds correctly

# --- Mapping Functions (Keep as before) ---
def map_predictions_to_gt_engine(y_pred_series):
    """ Maps detailed prediction strings to basic GT engine fault labels. """
    mapping = {
        'Normal_Predicted': 'Normal',
        'OilLeak_Predicted': 'OilLeak_Active',
        'BearingWear_Predicted': 'BearingWear_Active',
        'EGTSensorFail_Predicted': 'Normal', 
        'VibSensorFail_Predicted': 'Normal', 
        'Error_NoEvidence': 'Unknown',
        'Error_PipelineFail': 'Unknown',
        'Error_MissingInput': 'Unknown',
        'Error_FuncNotFound': 'Unknown'
    }
    return y_pred_series.map(mapping).fillna('Unknown') # Map NaNs or unexpected values

def map_predictions_to_gt_sensor(y_pred_series, sensor_type):
    """ Maps prediction strings relevant to sensor health back to GT labels. """
    if sensor_type == 'EGT':
        fail_pred = 'EGTSensorFail_Predicted'
        other_fail_pred = 'VibSensorFail_Predicted'
    elif sensor_type == 'Vibration':
        fail_pred = 'VibSensorFail_Predicted'
        other_fail_pred = 'EGTSensorFail_Predicted'
    else:
        return pd.Series(['Unknown'] * len(y_pred_series), index=y_pred_series.index)

    mapping = {
        fail_pred: 'Failed',
        other_fail_pred: 'OK', # If the *other* sensor failed, this one is OK
        'Normal_Predicted': 'OK',
        'OilLeak_Predicted': 'OK',
        'BearingWear_Predicted': 'OK',
        'Error_NoEvidence': 'Unknown',
        'Error_PipelineFail': 'Unknown',
        'Error_MissingInput': 'Unknown',
        'Error_FuncNotFound': 'Unknown'
    }
    return y_pred_series.map(mapping).fillna('Unknown')

# --- Metric Calculation Functions (Keep as before) ---
def calculate_metrics(y_true, y_pred, labels):
    """ Calculates accuracy, precision, recall, f1 (weighted). """
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    if not valid_indices.any():
        print("      Warning: No valid matching labels found.")
        return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    if len(y_true_filtered) == 0:
        print("      Warning: No overlapping valid labels after filtering.")
        return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan}
        
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=labels, average='weighted', zero_division=0)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

def generate_classification_report(y_true, y_pred, labels, target_names=None):
    """ Generates and returns the classification report string. """
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    if len(y_true_filtered) == 0: return "No valid matching labels for classification report."
    if target_names is None: target_names = labels
    present_labels = sorted(list(set(y_true_filtered) | set(y_pred_filtered)))
    present_target_names = [name for name, label in zip(target_names, labels) if label in present_labels]
    if not present_labels: return "No common labels found to generate report."
        
    report = classification_report(
        y_true_filtered, y_pred_filtered, labels=present_labels, target_names=present_target_names, zero_division=0)
    return report

def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix', save_path=None):
    """ Generates and optionally saves a confusion matrix plot. """
    valid_indices = y_true.isin(labels) & y_pred.isin(labels)
    y_true_filtered = y_true[valid_indices]
    y_pred_filtered = y_pred[valid_indices]
    if len(y_true_filtered) == 0:
        print(f"      Warning: No valid data for confusion matrix: {title}")
        return
        
    # Use only labels present in the filtered data for the plot axes
    present_labels_true = sorted(y_true_filtered.unique())
    present_labels_pred = sorted(y_pred_filtered.unique())
    all_present_labels = sorted(list(set(present_labels_true) | set(present_labels_pred)))
    
    # Ensure the labels argument to confusion_matrix contains all relevant labels
    cm_labels = [l for l in labels if l in all_present_labels]
    if not cm_labels:
        print(f"      Warning: No common labels found for confusion matrix: {title}")
        return

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=cm_labels)
    
    plt.figure(figsize=(len(cm_labels)*1.5 + 2, len(cm_labels)*1.5)) # Adjust size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150)
            print(f"      Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"      Error saving confusion matrix: {e}")
        finally:
             plt.close() # Ensure plot is closed
    else:
        plt.show()


# --- Main Analysis Execution Block ---
if __name__ == "__main__":
    print(f"--- Starting Evaluation Analysis ---")
    print(f"Loading results from: {RESULTS_FILE}")

    # Load the results data
    try:
        df_results = pd.read_csv(RESULTS_FILE)
        print(f"Loaded results data shape: {df_results.shape}")
        # Basic check for expected columns
        required_cols = ['Engine_Fault_State', 'EGT_Sensor_Health', 'Vibration_Sensor_Health'] + \
                    list(MODELS_TO_EVALUATE.values())
        if not all(col in df_results.columns for col in required_cols):
             print("Error: Results CSV is missing required columns.")
             print(f"Expected columns like: {required_cols}")
             print(f"Found columns: {df_results.columns.tolist()}")
             exit()
             
    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_FILE}")
        print("Please run run_experiments.py first.")
        exit()
    except Exception as e:
        print(f"Error loading results data: {e}")
        exit()

    # --- Perform Analysis for Each Model ---
    for model_name, pred_col in MODELS_TO_EVALUATE.items():
        print(f"\n{'='*10} Evaluating Model: {model_name} {'='*10}")

        y_pred_raw = df_results[pred_col]

        # --- Engine Fault Evaluation ---
        print("\n  --- Engine Fault Metrics ---")
        y_true_engine = df_results['Engine_Fault_State']
        y_pred_engine_mapped = map_predictions_to_gt_engine(y_pred_raw)
        
        engine_metrics = calculate_metrics(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT)
        print(f"    Overall Weighted Metrics: {engine_metrics}")
        
        engine_report = generate_classification_report(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT)
        print("\n    Classification Report:")
        print(engine_report)
        # Save report to file
        report_path = os.path.join(OUTPUT_DIR, f'report_engine_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Engine Fault Classification Report for Model: {model_name}\n")
            f.write("="*50 + "\n")
            f.write(engine_report)
            f.write("\n" + "="*50 + "\n")
            f.write(f"Overall Weighted Metrics: {engine_metrics}\n")
        print(f"    Engine report saved to {report_path}")

        # Plot confusion matrix
        cm_path = os.path.join(OUTPUT_DIR, f'cm_engine_{model_name}.png')
        plot_confusion_matrix(y_true_engine, y_pred_engine_mapped, LABELS_ENGINE_FAULT, 
                              title=f'Engine Fault CM ({model_name})', save_path=cm_path)


        # --- Sensor Health Evaluation (Only for applicable models) ---
        if model_name in MODELS_WITH_SENSOR_PREDS:
            # EGT Sensor Health
            print("\n  --- EGT Sensor Health Metrics ---")
            y_true_egt_health = df_results['EGT_Sensor_Health']
            y_pred_egt_health_mapped = map_predictions_to_gt_sensor(y_pred_raw, 'EGT')
            
            egt_health_metrics = calculate_metrics(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH)
            print(f"    Overall Weighted Metrics: {egt_health_metrics}")
            
            egt_health_report = generate_classification_report(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH)
            print("\n    Classification Report:")
            print(egt_health_report)
            report_path_egt = os.path.join(OUTPUT_DIR, f'report_egt_health_{model_name}.txt')
            with open(report_path_egt, 'w') as f:
                 f.write(f"EGT Sensor Health Classification Report for Model: {model_name}\n")
                 f.write("="*50 + "\n")
                 f.write(egt_health_report)
                 f.write("\n" + "="*50 + "\n")
                 f.write(f"Overall Weighted Metrics: {egt_health_metrics}\n")
            print(f"    EGT Health report saved to {report_path_egt}")

            cm_path_egt = os.path.join(OUTPUT_DIR, f'cm_egt_health_{model_name}.png')
            plot_confusion_matrix(y_true_egt_health, y_pred_egt_health_mapped, LABELS_SENSOR_HEALTH, 
                                  title=f'EGT Sensor Health CM ({model_name})', save_path=cm_path_egt)

            # Vibration Sensor Health
            print("\n  --- Vibration Sensor Health Metrics ---")
            y_true_vib_health = df_results['Vibration_Sensor_Health']
            y_pred_vib_health_mapped = map_predictions_to_gt_sensor(y_pred_raw, 'Vibration')
            
            vib_health_metrics = calculate_metrics(y_true_vib_health, y_pred_vib_health_mapped, LABELS_SENSOR_HEALTH)
            print(f"    Overall Weighted Metrics: {vib_health_metrics}")
            
            vib_health_report = generate_classification_report(y_true_vib_health, y_pred_vib_health_mapped, LABELS_SENSOR_HEALTH)
            print("\n    Classification Report:")
            print(vib_health_report)
            report_path_vib = os.path.join(OUTPUT_DIR, f'report_vib_health_{model_name}.txt')
            with open(report_path_vib, 'w') as f:
                 f.write(f"Vibration Sensor Health Classification Report for Model: {model_name}\n")
                 f.write("="*50 + "\n")
                 f.write(vib_health_report)
                 f.write("\n" + "="*50 + "\n")
                 f.write(f"Overall Weighted Metrics: {vib_health_metrics}\n")
            print(f"    Vibration Health report saved to {report_path_vib}")

            cm_path_vib = os.path.join(OUTPUT_DIR, f'cm_vib_health_{model_name}.png')
            plot_confusion_matrix(y_true_vib_health, y_pred_vib_health_mapped, LABELS_SENSOR_HEALTH, 
                                  title=f'Vibration Sensor Health CM ({model_name})', save_path=cm_path_vib)
        else:
             print(f"\n  --- Sensor Health Metrics (Skipped for {model_name}) ---")


    # --- (Optional) Add Per-Scenario Analysis Here ---
    # Loop through df_results['Scenario'].unique()
    # Filter df_results by scenario
    # Repeat metric calculations and saving for each scenario subset

    # --- (Optional) Add Robustness Metric Calculation Here ---
    # Filter df_results for Sensor Failure Scenarios
    # Compare F1 scores (or other metrics) for Full_DBN vs Vanilla_DBN on these subsets

    print(f"\n--- Evaluation Analysis Finished ---")
    print(f"Outputs saved in: {OUTPUT_DIR}")