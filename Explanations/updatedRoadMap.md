---

**Project Roadmap Status (Revised End of Week 3)**

**Focus:** Deliver a working **Integrated DBN + SimpleMRF system**, well-tested and evaluated on simulated data, with a clear report demonstrating component contributions.

---

**✅ Week 1 (Mar 30 - Apr 5): Foundations & Focused Data**

*   **Goal:** Set up, finalize scope, simulate and prepare the core dataset with ground truth.
*   **Status:** **COMPLETE.** Project set up, scope confirmed (`config.py`). `simulate_data.py` implemented and validated. `discretize_data.py` created `sim_data_discrete.csv`. Data saved with ground truth labels.

---

**✅ Week 2 (Apr 6 - Apr 12): DBN Core Build (Initial)**

*   **Goal:** Define the structure and initial probabilities for the DBN, initially focusing on component health affecting sensors.
*   **Status:** **COMPLETE.** `dbn_model.py` implemented. Initial DBN structure defined (nodes, edges). DBN visualized. Initial CPTs defined manually. Model structure validated. *(Did not yet include sensor health affecting observations)*.

---

**✅ Week 3 (Apr 13 - Apr 19): DBN Inference & Sensor Health Integration & Validation**

*   **Goal:** Get DBN inference running. **Integrate sensor health modeling directly into the DBN and validate the full integrated model.**
*   **Status:** **COMPLETE.**
    *   **DBN Structure & CPT Update:** Revised `dbn_model.py` structure to include edges from Sensor Health nodes to Observation nodes. Updated observation CPTs to depend on both physical health and sensor health. Included CPTs for sensor health nodes (initial and transition). Model re-validated.
    *   **DBN Inference:** `dbn_inference.py` implemented using `DBNInference`. Tested successfully by running inference on **all key scenarios** (Normal, OilLeak, BearingWear, EGTSensorFail, VibSensorFail).
    *   **Integrated Model Validation:** Generated plots showing inferred probabilities for *all* hidden states (component health + sensor health) for each scenario. **Validated** that the model:
        *   Correctly infers component health degradation in fault scenarios (e.g., Core Health in BearingWear).
        *   Correctly infers sensor health degradation in sensor failure scenarios (e.g., EGT Sensor Health in EGTSensorFail).
        *   Correctly distinguishes sensor failures from component failures (e.g., keeps Engine Core Health OK during sensor failures).

*   **Milestone:** Integrated DBN inference functional and **validated across all relevant scenarios**, demonstrating successful modeling of both component and sensor health within the unified DBN framework.

---

**➡️ Week 4 (Apr 20 - Apr 26): Simple MRF & Prediction Mapping**

*   **Goal:** Implement and validate the simple MRF smoother. Define prediction logic based on DBN outputs. Ensure full pipeline runs.
*   **Tasks:**
    *   **Simple MRF Implementation & Validation:**
        *   Implement MRF smoother for raw Vib1/Vib2 in `mrf_model.py` (or `utils.py`).
        *   Test smoother on noisy raw simulation data. Plot raw vs. smoothed vibration data. Visually confirm smoothing/consistency effect.
    *   **Define Prediction Logic:** Decide and document probability threshold(s) to convert DBN output probabilities (P(CoreHealth='Warn'/'Fail'), P(LubHealth='Fail'), P(EGT_SH='Failed'), P(Vib_SH='Failed')) into discrete final class predictions ('Normal', 'Oil Leak', 'Bearing Wear', 'EGT Sensor Failure', 'Vibration Sensor Failure'). Add this logic (e.g., in `evaluation.py` or `utils.py`).
    *   **Pipeline Integration (V1):** Update or create `main_script.py` to run the full sequence: **MRF (on raw Vib) -> Discretization (incl. smoothed Vib) -> DBN Inference (on discrete data) -> Prediction Logic**.
    *   **Test Pipeline Flow:** Debug end-to-end data flow on a short sequence, ensuring data passes correctly and final discrete predictions are generated.
*   **Milestone:** Simple MRF implemented and validated. Prediction logic defined. Full pipeline runs end-to-end generating discrete predictions.

---

**➡️ Week 5 (Apr 27 - May 3): Evaluation Setup & Experiments**

*   **Goal:** Implement evaluation metrics. Evaluate the full system against baselines using systematic experiments.
*   **Tasks:**
    *   **Baseline Implementation:** Code rule-based and **Vanilla DBN (without sensor health nodes/dependencies or MRF)** baselines for comparison.
    *   **Metrics Implementation:** Implement `evaluation.py` with functions for Confusion Matrix, Accuracy, Precision, Recall, F1-Score (using `scikit-learn`) comparing model predictions against **ground truth `Engine_Fault_State` and ground truth `*_Sensor_Health` labels**.
    *   **Run Experiments:** For **all scenarios**: Generate multiple simulation runs. Run **Full System (Integrated DBN + MRF)**, **Vanilla DBN**, **Rule-based**. Apply prediction logic.
    *   **Collect & Organize Results:** Store ground truth labels, model predictions, and calculated metrics.
*   **Milestone:** All experiments completed. Raw predictions and calculated metrics collected.

---

**➡️ Week 6 (May 4 - May 10): Analysis, Report & Code Polish**

*   **Goal:** Analyze results comprehensively, write the report draft incorporating evaluation, clean and document code.
*   **Tasks:**
    *   **Analyze Results & Metrics:** Calculate average metrics. Generate Confusion Matrices. Calculate **Robustness Metrics:** Specifically compare Full System vs. Vanilla DBN performance on the **Sensor Failure scenarios** to quantify the benefit of the **integrated sensor health modeling**. Analyze MRF impact separately if possible.
    *   **Create Results Visuals:** Final comparison tables, CM plots, MRF validation plots, DBN probability timelines.
    *   **Write Report Draft:** Describe scope, models (Integrated DBN, MRF), evaluation. Present quantitative results. Discuss results, explicitly quantifying the **impact/benefit of integrated sensor health modeling and MRF** based on metrics. Include Ethics, Limits, Robustness discussion.
    *   **Code Cleaning & Documentation.**
*   **Milestone:** Report drafted with comprehensive analysis. Figures/tables generated. Code cleaned.

---

**➡️ Final Days (May 11 - May 12): Final Review & Submission**

*   **Goal:** Submit a polished, well-evaluated project on time.
*   *(Tasks as originally planned)*
