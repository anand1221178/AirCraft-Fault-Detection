**Understanding the Goal:**

You've created *simulated* data. Think of it like a flight simulator, but instead of simulating the plane's flight, you're simulating the engine's sensors under different conditions (normal flight, having an oil leak, having bearing wear, or having a sensor itself break). The goal is to create realistic-looking data *where you already know the answers* (the ground truth) so you can later test if your diagnostic model (the DBN+HMM+MRF) can figure out the correct answers just by looking at the sensor readings.

**1. What the CSV Data Means (Column by Column):**

Let's look at the header you provided:

`Timestamp, EGT_C, N2_PctRPM, OilPressure_PSI, Vib1_IPS, Vib2_IPS, Engine_Fault_State, EGT_Sensor_Health, Vibration_Sensor_Health, Scenario`

*   **`Timestamp`**: The exact date and time when this row of sensor readings was recorded (or simulated). Crucial for seeing how things change over time.
*   **`EGT_C`**: Exhaust Gas Temperature in degrees Celsius. A measure of how hot the engine is running internally. High values can indicate high load or potential inefficiency/problems.
*   **`N2_PctRPM`**: Speed of the engine's high-pressure core shaft, as a percentage of its maximum RPM. Indicates how much power the core is generating.
*   **`OilPressure_PSI`**: Lubrication system pressure in Pounds per Square Inch. Critical for engine health; low pressure usually means an oil leak or pump problem.
*   **`Vib1_IPS` / `Vib2_IPS`**: Vibration measured by two different sensors (Sensor 1 and Sensor 2) in Inches Per Second. High vibration usually points to mechanical problems like imbalance or wear (like bearing wear). They should generally read similar values.
*   **`Engine_Fault_State`**: **This is a Ground Truth label.** It tells you the *actual, real* condition of the engine *you simulated* at this specific timestamp. It's the "answer" your model needs to learn to predict. Possible values here are:
    *   `Normal`: The engine is operating correctly.
    *   `OilLeak_Active`: The simulated oil leak is currently happening.
    *   `BearingWear_Active`: The simulated bearing wear is currently happening.
*   **`EGT_Sensor_Health`**: **Another Ground Truth label.** It tells you the *actual* state of the EGT sensor itself at this timestamp. Is the *sensor* working correctly? Possible values:
    *   `OK`: The sensor is working fine.
    *   `Degraded`: The sensor is starting to fail (e.g., drifting, noisy).
    *   `Failed`: The sensor is giving very unreliable readings.
*   **`Vibration_Sensor_Health`**: **Ground Truth label** for the vibration sensors (treated as a system). Tells you if the vibration sensors themselves are working correctly ('OK', 'Degraded', 'Failed').
*   **`Scenario`**: Tells you which overall simulation run this row belongs to (e.g., 'Normal', 'OilLeak', 'BearingWear', 'EGTSensorFail', 'VibSensorFail'). This helps organize the data.

**Example CSV Row Interpretation:**

Let's take your first sample row from the 'BearingWear' scenario:

`2025-04-02 10:04:37.739700,,88.82,56.41,2.03,2.05,BearingWear_Active,OK,OK,BearingWear`

*   **When:** April 2nd, 2025, at about 10:04:37 AM.
*   **EGT:** The value is missing (`, ,`). This represents a **sensor dropout** – a moment where the EGT sensor didn't provide a reading (simulated using `NaN`).
*   **N2:** Engine core speed is 88.82 %RPM (within the normal cruise range).
*   **Oil Pressure:** 56.41 PSI (also normal).
*   **Vibration:** Vib1 is 2.03 IPS, Vib2 is 2.05 IPS. These are **high** (above the 1.5 IPS threshold) and very similar to each other.
*   **Engine State (Truth):** The engine *actually* has `BearingWear_Active`.
*   **Sensor Health (Truth):** Both the EGT sensor (`OK`) and the Vibration sensors (`OK`) are working correctly at this moment.
*   **Scenario:** This data comes from the 'BearingWear' simulation run.

**Meaning:** At this specific second, the simulation shows an engine suffering from bearing wear. This wear is causing high vibration readings (Vib1/Vib2 are high). Even though the engine has a problem, the sensors *reporting* the vibration are working fine. The oil pressure and N2 speed are currently normal. The EGT sensor happened to have a dropout at this exact moment.

**2. What the Plots Mean:**

The plots you generated are visual summaries of the CSV data over time for each specific scenario. They are essential for **validation** – checking that the simulation code is actually doing what you intended.

*   **X-axis (Shared):** Time (`Timestamp`).
*   **Y-axis (Individual Plots):** The value of a specific sensor (EGT, N2, Oil Pressure, Vibration).
*   **Blue/Orange Lines:** These are the actual simulated sensor readings over time (the values from columns like `EGT_C`, `N2_PctRPM`, `Vib1_IPS`, etc.). You can see how they fluctuate due to noise and how they change when a fault or sensor failure occurs. The dots show individual readings, highlighting dropouts (gaps).
*   **Dotted Horizontal Lines (Red/Orange):** These are the **thresholds** you defined (e.g., `EGT_low_thresh`, `EGT_high_thresh`). They represent the boundaries you'll likely use later to categorize the continuous sensor values into discrete states like 'Low', 'Medium', 'High'. They help you visually judge if a sensor reading is normal or abnormal.
*   **Purple Line (on secondary Y-axis):** This shows the **Ground Truth Sensor Health** ('OK', 'Degraded', 'Failed') over time for the sensors that have an HMM applied (EGT and Vibration). You can see exactly when the simulation made the sensor start to fail.
*   **Vertical Dashed Lines (Black/Magenta - added by the plotting code):** These mark the *onset time* of the simulated fault (like 'OilLeak' starting) or the sensor failure ('EGTSensorFail' starting).

**Plot Interpretation Examples:**

*   **OilLeak Plot:** Look at the 'OilPressure_PSI' subplot. You see the blue line staying steady around 55 PSI (Medium), then after the fault onset time (marked by the dashed line), it drops sharply down towards and below the 'Low' threshold (40 PSI). The other sensor plots (EGT, N2, Vib) remain steady. This visually confirms: "An oil leak primarily causes oil pressure to drop."
*   **BearingWear Plot:** Look at the 'Vibration (IPS)' subplot. Both orange and blue lines start around 1.0 IPS (Medium), then after the fault onset, they rise together above the 'High' threshold (1.5 IPS). Now look at the 'EGT_C' subplot. You see the EGT line also drifts slightly upwards, towards its 'High' threshold (950°C), around the same time the vibration increases. This visually confirms: "Bearing wear causes high vibration (primary effect) and slightly increased EGT (secondary effect)."
*   **EGTSensorFail Plot:** Look at the 'EGT_C' subplot. The blue line is normal initially, but after the sensor failure onset time, it suddenly drifts way up (or down, or gets very noisy), going outside the normal thresholds. *Crucially*, look at the purple line on the secondary axis – it drops from 'OK' to 'Degraded'/'Failed' exactly when the blue EGT line goes crazy. Also, notice that the *other* sensor plots (N2, OilP, Vib) look perfectly normal during this time. This visually confirms: "The EGT *sensor* failed, giving bad readings, but the *engine* itself was fine."

**In essence:** The CSV file is the raw material containing the moment-by-moment simulated reality, including the hidden "ground truth". The plots are the visual reports that help you confirm the simulation created the specific patterns and events you intended for each scenario, matching the descriptions in your project plan.