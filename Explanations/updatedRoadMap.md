
# Project Roadmap Status (Revised – No HMM Version)

## ✅ Week 1 (Mar 30 - Apr 5): Foundations & Focused Data

**Goal:** Set up, finalize scope, simulate and prepare the core dataset with ground truth.  
**Status:** ✅ COMPLETE  
- `simulate_data.py` implemented and validated.  
- `discretize_data.py` created `sim_data_discrete.csv`.  
- Data saved with ground truth labels.

---

## ✅ Week 2 (Apr 6 - Apr 12): DBN Core Build (Initial)

**Goal:** Define the structure and initial probabilities for the DBN, initially focusing on component health affecting sensors.  
**Status:** ✅ COMPLETE  
- `dbn_model.py` implemented: nodes, edges, CPTs.  
- Initial DBN validated.  
- Sensor health and reliability modeling planned for integration.

---

## ✅ Week 3 (Apr 13 - Apr 19): Integrated DBN Inference with Sensor Health

**Goal:** Implement DBN inference and extend model to include sensor health nodes affecting observations.  
**Status:** ✅ COMPLETE  
- `dbn_model.py` updated with Sensor Health → Observation edges.  
- CPTs updated accordingly.  
- `dbn_inference.py` built and tested across all scenarios.  
- Generated and validated inference plots for all key hidden variables.  
- DBN now models both component and sensor degradation **without HMMs**.

---

## ✅ Week 4 (Apr 20 - Apr 26): Simple MRF & Prediction Logic

**Goal:** Add simple MRF smoother for Vib1/Vib2 and define prediction logic for faults.  
**Status:** ✅ COMPLETE  
- MRF applied to pre-process vibration signals.  
- Smoother validated visually.  
- Defined threshold-based prediction logic.  
- Full pipeline integrated (MRF → Discretize → DBN → Prediction).

---

## ✅ Week 5 (Apr 27 - May 3): Evaluation Setup & Experiments

**Goal:** Implement evaluation metrics, run systematic experiments.  
**Status:** ✅ COMPLETE  
- Vanilla DBN and Rule-based baselines implemented.  
- `evaluation.py` computes confusion matrix, accuracy, precision, recall, F1.  
- Experiments executed on all scenarios.  
- Results saved for Full DBN, Vanilla DBN, Rule-Based classifiers.

---

## 🔄 Week 6 (May 4 - May 10): Results Analysis, GUI & Report

**Goal:** Analyze metrics, polish results and generate GUI/report.  
**Status:** ✅ IN PROGRESS  
- Full DBN shows significant robustness over baselines.  
- Flask GUI created to render evaluation metrics and plots.  
- Classification reports converted to HTML tables.  
- Final visualizations under development.  
- Report writing ongoing.

---

## ⏳ Final Days (May 11 - May 12): Final Polish & Submission

**Goal:** Finalize report, clean project, and submit.  
**Tasks:**  
- Final edits to report and visualizations  
- Code documentation and file cleanup  
- Submission packaging
