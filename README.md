# ✈️ Aircraft Engine Fault Detection using PGMs

This project implements a **Dynamic Bayesian Network (DBN)** integrated with a **Markov Random Field (MRF)** to detect early-stage faults in aircraft engines. The system processes noisy, discretized sensor data and distinguishes between mechanical degradation and sensor failures.

---

## 🧠 Core Ideas

- **DBN** captures temporal dependencies between engine subsystem health and sensor observations.
- **MRF** smooths inconsistent vibration readings from colocated sensors (Vib1/Vib2).
- **Sensor Health Modeling** enables robustness to sensor drift and failure.
- **Simulated Dataset** mimics five realistic operational scenarios:
  - Normal Operation
  - Oil Leak
  - Bearing Wear
  - EGT Sensor Failure
  - Vibration Sensor Failure

---

## 📁 Project Structure

| Folder/File        | Purpose |
|--------------------|---------|
| `Data_Gen/`        | Sensor simulation & discretization logic |
| `DBN/`             | DBN model creation & inference implementation |
| `PreProcessing/`   | MRF-based vibration smoothing |
| `Utils/`           | Evaluation scripts & rule-based baselines |
| `Data/`            | Generated data, plots, and model outputs |
| `PGM_GUI_Viewer/`  | Lightweight dashboard to explore results |
| `run_experiment.py`| Main pipeline: sim → DBN → classify |
| `evaluation.py`    | Outputs accuracy, F1, confusion matrices |
| `2561034_report.pdf` | Full technical write-up (IEEE format) |

---

## 🚀 How to Run

### 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate aircraft_env
```

### 2. Run the Experiment

```bash
python run_experiment.py
```

### 3. Evaluate Model Predictions

```bash
cd Utils/
python evaluation.py
```

### 4. View the Results in Browser (optional)

```bash
cd PGM_GUI_Viewer/
python viewer_app.py
```

Then open `http://127.0.0.1:5000` in your browser.

---

## 📊 Sample Output

- Accuracy: 85.1% (Full DBN + MRF)
- F1 Score: 0.796 (better than Vanilla DBN and Rule-Based)
- Robust under sensor failure scenarios
- Interpretable inference timelines & confusion matrices

---

## 📄 Report

See [`2561034_report.pdf`](2561034_report.pdf) for full methodology, results, and analysis.

---

## 🛠️ Future Work

- Structural learning from real engine data
- HMM/Kalman hybrid modules for continuous modeling
- Operator-in-the-loop feedback for adaptive thresholds

---

Built with ❤️ for Probabilistic Graphical Models coursework.
