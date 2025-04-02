1.  **BearingWear Plot:**
    *   ✅ **Vibration (Primary):** Vib1 & Vib2 clearly increase from the baseline (~1.0 IPS) towards and above the High threshold (1.5 IPS) after the fault onset time. They track each other well, showing correlation.
    *   ✅ **EGT (Secondary):** Shows a noticeable, correlated increase from its baseline (~900°C) towards the High threshold (950°C) as the vibration increases.
    *   ✅ **N2 & Oil Pressure:** Remain stable around their normal operating points, as expected.
    *   ✅ **Sensor Health:** Both EGT and Vibration sensor health ground truths remain 'OK'.

2.  **EGTSensorFail Plot:**
    *   ✅ **EGT:** Drifts significantly upwards *after* the sensor failure onset time, becoming much noisier and exceeding normal limits, *while other sensors remain normal*.
    *   ✅ **EGT Sensor Health (GT):** Correctly shows the transition from 'OK' -> 'Degraded' -> 'Failed' during the period the EGT reading is anomalous.
    *   ✅ **N2, Oil Pressure, Vibration:** Remain stable, indicating the engine itself is fine.
    *   ✅ **Vibration Sensor Health (GT):** Remains 'OK'.

3.  **Normal Plot:**
    *   ✅ **All Sensors:** Fluctuate with noise around their typical 'Medium' cruise values (EGT ~900, N2 ~90, OilP ~55, Vib ~1.0) throughout the run. No significant drifts or excursions.
    *   ✅ **Sensor Health:** Both EGT and Vibration sensor health ground truths remain 'OK'.

4.  **OilLeak Plot:**
    *   ✅ **Oil Pressure (Primary):** Shows a clear, significant drop from its normal level (~55 PSI) down towards the Low threshold (40 PSI) and below after the fault onset time.
    *   ✅ **EGT, N2, Vibration:** Remain stable around their normal operating points, as expected for the primary signature of this fault.
    *   ✅ **Sensor Health:** Both EGT and Vibration sensor health ground truths remain 'OK'.

5.  **VibSensorFail Plot:**
    *   ✅ **Vibration:** Vib1 & Vib2 drift significantly upwards *after* the sensor failure onset time, exceeding the High threshold, *while other sensors remain normal*.
    *   ✅ **Vibration Sensor Health (GT):** Correctly shows the transition from 'OK' -> 'Degraded' -> 'Failed' during the period the Vibration readings are anomalous.
    *   ✅ **EGT, N2, Oil Pressure:** Remain stable, indicating the engine itself is fine.
    *   ✅ **EGT Sensor Health (GT):** Remains 'OK'.

**In summary:** The plots demonstrate that your simulation correctly captures:
*   Normal baseline behavior.
*   The distinct primary and secondary sensor signatures for both Oil Leak and Bearing Wear faults.
*   Sensor failure modes where the sensor reading diverges significantly while the underlying system state (and other sensors) remain normal.
*   Accurate ground truth labeling for both engine fault state (implicit in the scenario plots) and sensor health states.
*   Appropriate levels of noise and the presence of dropouts (visible as small gaps/missing points in the traces).
