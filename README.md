# ✈️ Aircraft Engine Fault Detection using Dynamic Bayesian Networks

This project implements a **Dynamic Bayesian Network (DBN)** integrated with a **Markov Random Field (MRF)** to detect early-stage faults in aircraft engines. The model is trained and evaluated on the **NASA C-MAPSS FD001** dataset, using discretized sensor readings to classify health states: **Healthy**, **Degrading**, and **Critical**.

---

## 🧠 Project Highlights

- **DBN** models temporal transitions of engine health using multivariate sensor emissions.
- **MRF Smoothing** improves robustness to vibration noise by enforcing temporal consistency.
- **Sensor Binning Strategy** includes quantile, k-means, and uniform schemes across 9 selected sensors.
- **Threshold Optimization** via grid search for:
  - Accuracy-tuned
  - Macro-F1-tuned
  - Fixed threshold baselines
- **Extensive Evaluation** includes confusion matrices, per-class F1, posteriors, and error analysis.

---

## 📁 Project Structure

| Folder/File             | Purpose |
|--------------------------|---------|
| `DBN/`                   | DBN model structure, CPT learning, and inference |
| `Data_Gen/`              | Sensor binning and data preprocessing |
| `PreProcessing/`         | MRF smoothing for vibration channels |
| `Utils/`                 | Evaluation, thresholding, and plotting tools |
| `Data/plots/`            | Output CSVs and final figures for report |
| `run_experiment.py`      | Main pipeline: binning → DBN → inference → export |
| `evaluation.py`          | Confusion matrix, classification metrics |
| `2561034_report.pdf`     | Final 5-page IEEE report |

---

## ⚙️ How to Run the Pipeline

### 1. Setup Environment

```bash
conda create -n aircraft_env python=3.10
conda activate aircraft_env
pip install -r requirements.txt
```

Optional: or use the provided `environment.yaml`.
You also will have to run `pip install --upgrade "pgmpy<0.2.0"`
### 2. Run the Main Experiment

```bash
python run_experiment.py
```

This will:
- Discretize sensor data
- Smooth vibration sensors (optional)
- Learn DBN CPDs (MLE + smoothing)
- Run inference
- Save results to `Data/plots/`:
  - `unit1_posteriors.csv`
  - `all_probs_macroF1.csv`
  - `DBN_learned_fold{0..9}.csv`

---

## 📊 Generated Outputs

| Output File | Description |
|-------------|-------------|
| `unit1_posteriors.csv` | Per-cycle posteriors for a representative engine |
| `all_probs_macroF1.csv` | Final macro-F1 optimized posteriors and predictions |
| `DBN_learned_fold*.csv` | 10-fold cross-validation results |
| `confusion_macroF1.png` | Final confusion matrix plot |
| `learning_curve_f1breakdown.png` | Macro-F1 vs folds + per-class F1 bars |
| `error_breakdown_per_class.png` | FP and FN rates per class |
| `threshold_heatmap.png` | Macro-F1 score across grid of threshold values |

---

## 🧪 Metrics Summary

- **Accuracy (best tuned)**: 0.65  
- **Macro-F1 (best tuned)**: 0.6347  
- **Critical Recall**: > 0.80  
- **Degrading** is most difficult due to transitional ambiguity

---

## 📄 Report

All methodology, analysis, and results are presented in:  
📄 [`2561034_report.pdf`](2561034_report.pdf)

---

## 🧭 Suggested Future Work

- Learn DBN structure (not only CPDs)
- Integrate Kalman filter for hybrid continuous-discrete tracking
- Online CPT updates for drift adaptation
- Incorporate causal discovery and intervention modeling

---

Made for the **Probabilistic Graphical Models** final project  
🛫 Wits CS Honours, 2025
