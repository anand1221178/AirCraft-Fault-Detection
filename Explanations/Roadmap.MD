
**Focus:** Deliver a working DBN+HMM+SimpleMRF system, **well-tested and evaluated** on simulated data, with a clear report demonstrating component contributions.

---

**✅ Week 1 (Mar 30 - Apr 5): Foundations & Focused Data**

*   **Goal:** Set up, finalize the simplified scope, simulate and prepare the core dataset with ground truth.
*   **Tasks:**
    *   **Setup:** Project structure, Git, virtual env, install libraries (`pandas`, `numpy`, `matplotlib`, `pgmpy`, `networkx`, `scikit-learn` for metrics).
    *   **Finalize Scope:** Confirm 4 sensors (EGT, N2, Oil Pressure, Vibration x2), 2 fault types (Oil Leak, Bearing Wear), 2 HMM sensors (Vibration, EGT), MRF sensor pair (Vibration 1&2). Update `config.py`.
    *   **Implement Data Simulation:** Code `simulate_data.py` for the 4 sensors, normal operation, Oil Leak, Bearing Wear scenarios, **and a Sensor Failure scenario** (e.g., Vib or EGT fails mid-run during normal operation). Add noise/dropout. **Crucially, include accurate time-step level ground truth labels** (e.g., 'Normal', 'OilLeak_Active', 'BearingWear_Active', 'Normal_VibSensorFail').
    *   **Validate Simulation:** Generate data, plot sensor traces and ground truth labels for each scenario. Verify realism.
    *   **Implement Discretization:** Code `discretize_data.py` for 3 bins ('Low', 'Medium', 'High'). Test it.
    *   **Save Data:** Store simulated raw and discretized data (`sim_data_raw.csv`, `sim_data_discrete.csv`), including ground truth labels.
*   **Milestone:** Working data generation (including sensor failure scenario and ground truth labels) and discretization for the simplified scope.

---

**✅ Week 2 (Apr 6 - Apr 12): DBN Core Build**

*   **Goal:** Define the structure and initial probabilities for the simplified DBN.
*   **Tasks:**
    *   **DBN Structure Definition:** Define nodes and edges in `dbn_model.py`.
    *   **Visualize DBN:** Generate and save the DBN structure diagram.
    *   **CPT Definition (Simplified):** Define `TabularCPD`s for all nodes.
    *   **Check Model:** Ensure `dbn.check_model()` passes.
*   **Milestone:** DBN structure coded and visualized, initial CPTs defined, model structure is valid.

---

**✅ Week 3 (Apr 13 - Apr 19): DBN Inference & HMM Implementation & Validation**

*   **Goal:** Get basic DBN inference running. Implement and **validate** the two required HMMs.
*   **Tasks:**
    *   **DBN Inference:** Implement `run_dbn_inference` function (e.g., using `VariableElimination`). Test on short sequences (without HMM adjustments yet). Check if inferred health states make basic sense.
    *   **HMM Implementation (Vibration & EGT):** Implement HMMs in `hmm_model.py` (states, transitions, emissions, `estimate_sensor_health` function).
    *   **HMM Component Validation:**
        *   Generate specific simulation data where Vib or EGT sensor degrades/fails.
        *   Run the corresponding HMM on this data.
        *   **Plot estimated health states vs. known simulated ground truth sensor health.** Visually confirm HMM can detect the failure/degradation pattern. (Informal metric: visual confirmation).
*   **Milestone:** Basic DBN inference functional. HMMs implemented and **individually validated** against simulated sensor failure data.

---

**✅ Week 4 (Apr 20 - Apr 26): Simple MRF, Integration & Prediction Mapping**

*   **Goal:** Implement and validate the simple MRF smoother. Integrate HMM outputs into the DBN pipeline. Define prediction logic.
*   **Tasks:**
    *   **Simple MRF Implementation & Validation:**
        *   Implement MRF for Vib1/Vib2 in `mrf_model.py`.
        *   Test smoother on noisy simulation data. **Plot raw vs. smoothed vibration data.** Visually confirm basic smoothing/consistency effect. (Informal metric: visual confirmation).
    *   **Implement Evidence Adjustment:** Code `adjust_evidence_based_on_health` in `utils.py`.
    *   **Define Prediction Logic:** Decide and document the **probability threshold(s)** needed to convert DBN output probabilities (e.g., P(Lubrication='Fail')) into discrete class predictions ('Normal', 'Oil Leak', 'Bearing Wear'). Add this logic possibly to `evaluation.py` or `utils.py`.
    *   **Pipeline Integration (V1):** Update `run_full_pipeline` in `main_script.py` to include MRF, Discretization, HMMs, Evidence Adjustment, and DBN Inference steps.
    *   **Test Pipeline Flow:** Debug end-to-end data flow on a short sequence.
*   **Milestone:** Simple MRF implemented and validated. HMM output integrated. Prediction threshold defined. Full pipeline runs end-to-end.

---

**✅ Week 5 (Apr 27 - May 3): Evaluation Setup & Experiments**

*   **Goal:** Implement evaluation metrics. Evaluate the full system against baselines using systematic experiments.
*   **Tasks:**
    *   **Baseline Implementation:** Code rule-based and Vanilla DBN baselines.
    *   **Metrics Implementation:** Implement `evaluation.py` with functions to calculate: **Confusion Matrix, Accuracy, Precision (per class), Recall (per class), F1-Score (per class & weighted average)** using `scikit-learn`.
    *   **Run Experiments:** For **all scenarios (Normal, Oil Leak, Bearing Wear, Sensor Failure)**:
        *   Generate multiple simulation runs (N=10-20).
        *   Run **Full System (A)**, **Vanilla DBN (B)**, **Rule-based (C)**.
        *   Apply prediction logic to get discrete predictions for each model.
    *   **Collect & Organize Results:** Store **ground truth labels, model predictions**, and calculated **metrics** for all runs/models/scenarios (`experiment_results.csv` or similar).
*   **Milestone:** All experiments completed for all models and scenarios, including the sensor failure scenario. Raw predictions and calculated metrics collected and organized.

---

**✅ Week 6 (May 4 - May 10): Analysis, Report & Code Polish**

*   **Goal:** Analyze results comprehensively, write the report draft incorporating evaluation, clean and document code.
*   **Tasks:**
    *   **Analyze Results & Metrics:**
        *   Calculate average metrics and standard deviations.
        *   Generate **Confusion Matrices** for key models/scenarios.
        *   Calculate **Robustness Metrics:** Compare Full System vs. Vanilla DBN performance (e.g., % change in F1) overall and specifically on the **Sensor Failure scenario**.
    *   **Create Results Visuals:** Produce final comparison table (Avg F1/P/R), Confusion Matrix plots, HMM/MRF validation plots, potentially DBN probability timelines.
    *   **Write Report Draft:**
        *   Describe simplified scope, models, and **evaluation methodology/metrics**.
        *   Present **quantitative results** (tables, metrics).
        *   Discuss the results, explicitly comparing models and quantifying the **impact/benefit of HMMs and MRF** based on the analysis (especially sensor failure scenario results).
        *   Include Ethics, Limitations, **Robustness discussion informed by metrics**.
        *   Ensure 5-page limit.
    *   **Code Cleaning & Documentation:** Comments, README.md, `requirements.txt`.
*   **Milestone:** Report drafted with comprehensive results analysis and discussion based on implemented metrics. Figures/tables generated. Code cleaned/documented.

---

**✅ Final Days (May 11 - May 12): Final Review & Submission**

*   **Goal:** Submit a polished, well-evaluated project on time.
*   **Tasks:**
    *   **Proofread & Format Report:** Final checks.
    *   **Final Code Check:** Ensure reproducibility.
    *   **Package:** Create final ZIP archive.
    *   **Submit!**
*   **Milestone:** Project Submitted!

