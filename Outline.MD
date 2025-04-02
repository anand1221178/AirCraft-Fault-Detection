**1. The Core Problem**

We want to detect potential issues in an aircraft engine *before* they become critical, using limited sensor data that might be noisy or unreliable. We need a system that understands:
a) How different components (like the lubrication system or core rotating parts) affect sensor readings.
b) How component health changes over time.
c) That sensors themselves might not be perfectly reliable.
d) That related sensors should generally provide consistent information.

**2. The Goal**

To build a probabilistic model (DBN + HMM + simple MRF) that takes readings from 4 key sensors, accounts for potential sensor degradation (for 2 sensors) and localized noise (for 1 sensor type), and outputs the probability of two specific fault conditions ("Oil Leak", "Bearing Wear") and the overall health state of the engine core and lubrication system.

**3. The Sensors**

We'll use 3 discretization levels: **Low (L)**, **Medium (M)**, **High (H)**. The exact numerical ranges are *highly* dependent on the specific engine and operating phase (takeoff, cruise, descent, idle). For this project, we *define* 'Medium' as the typical range during stable cruise conditions.

*   **a) EGT (Exhaust Gas Temperature)**
    *   *What it measures:* Temperature of exhaust gases after the turbine. Proxy for thermal load and combustion efficiency.
    *   *Typical States (Relative to Cruise):*
        *   **L:** Idle, descent, potentially very efficient low-power cruise. (< 850°C example)
        *   **M:** Normal cruise, stable climb. (850°C - 950°C example)
        *   **H:** Takeoff, high-power climb, potentially indicative of engine inefficiency, cooling problems, or early stages of certain faults. (> 950°C example, with critical limits much higher).
    *   *Fault Correlations:*
        *   **Oil Leak:** Minimal *direct* correlation. A severe, prolonged leak leading to friction might *eventually* raise EGT, but it's a late/secondary effect.
        *   **Bearing Wear:** Moderate positive correlation. Increased friction/inefficiency from worn bearings can raise the overall core temperature, pushing EGT from 'M' towards 'H', especially under higher load.
    *   *Other Correlations:* Positively correlated with N2 speed and fuel flow.
    *   *PGM Role:* Key evidence for `Engine_Core_Health` in DBN. **HMM applied** to model its potential drift/degradation, making its reading conditionally reliable for the DBN.

*   **b) N2 (High-Pressure Spool Speed)**
    *   *What it measures:* Rotational speed (% of max RPM) of the engine core (high-pressure compressor/turbine). Reflects core power generation.
    *   *Typical States (Relative to Cruise):*
        *   **L:** Idle, descent. (< 85% RPM example)
        *   **M:** Normal cruise, stable climb. (85% - 95% RPM example)
        *   **H:** Takeoff, high-power climb. (> 95% RPM example)
    *   *Fault Correlations:*
        *   **Oil Leak:** Minimal direct correlation, unless catastrophic failure occurs.
        *   **Bearing Wear:** Weak negative correlation *possible*. Increased drag *might* require slightly more fuel to maintain the same N2, or result in slightly lower N2 for the same fuel flow. This is often subtle.
    *   *Other Correlations:* Strongly positively correlated with EGT, Fuel Flow, P3 pressure.
    *   *PGM Role:* Key evidence for `Engine_Core_Health` and overall engine state in DBN. **No HMM applied** in this simplified scope (assumed reliable).

*   **c) Oil Pressure**
    *   *What it measures:* Pressure within the lubrication system. Critical for preventing component damage.
    *   *Typical States (Relative to Cruise):*
        *   **L:** Dangerously low pressure. Indicates leak, pump failure, or very low oil level. (< 40 PSI example)
        *   **M:** Normal operating range. (40 - 70 PSI example)
        *   **H:** Less common, potentially cold oil, relief valve issue. ( > 70 PSI example - might merge with 'M' if only 3 bins and high isn't a fault symptom here).
    *   *Fault Correlations:*
        *   **Oil Leak:** **Strong negative correlation.** This is the primary indicator. Causes pressure to drop from 'M' to 'L'.
        *   **Bearing Wear:** Minimal direct correlation in this simplified scope.
    *   *Other Correlations:* Weak negative correlation with Oil Temperature (hotter = thinner = lower pressure). Weak positive correlation with N2 (drives the oil pump).
    *   *PGM Role:* Primary evidence for `Lubrication_System_Health` in DBN. **No HMM applied** (we assume its failure mode is mainly catastrophic signal loss, not drift, for simplicity here, although real sensors can fail complexly).

*   **d) Vibration (Sensor 1 & Sensor 2 - for MRF)**
    *   *What it measures:* Engine casing acceleration/movement, indicating mechanical imbalance or roughness. Assume Vib1 and Vib2 are placed to monitor core rotational components.
    *   *Typical States (Relative to Cruise):*
        *   **L:** Very smooth operation. (< 0.5 IPS example)
        *   **M:** Normal operational vibration. (0.5 - 1.5 IPS example)
        *   **H:** High vibration, indicates mechanical issue. (> 1.5 IPS example)
    *   *Fault Correlations:*
        *   **Oil Leak:** Minimal direct correlation until potential secondary lubrication failure causes damage.
        *   **Bearing Wear:** **Strong positive correlation.** This is the primary indicator. Causes vibration to increase from 'L'/'M' towards 'H'.
    *   *Other Correlations:* Vib1 and Vib2 *should be strongly correlated* with each other if monitoring the same phenomenon. Weak positive correlation with N2 (higher speed *can* slightly increase baseline vibration).
    *   *PGM Role:* Primary evidence for `Engine_Core_Health` (specifically mechanical integrity) in DBN. **HMM applied** (to the *concept* of core vibration, maybe using the smoothed value) because these sensors are prone to noise/drift. **Simple MRF applied** to Vib1 and Vib2 readings *before* discretization/HMM/DBN to enforce consistency and reduce spurious noise spikes between the two physically close sensors.

**4. The Faults (Our 2 Focus Areas)**

*   **a) Oil Leak**
    *   *In-depth:* A breach in the lubrication system circuit. Can be slow or fast. Consequences range from requiring oil top-ups to complete loss of pressure, oil starvation of bearings/gears, rapid overheating due to friction, and potentially catastrophic engine seizure or fire.
    *   *Sensor Signature (Simplified Scope):*
        *   `Oil Pressure`: **Primary.** Drops from 'M' -> 'L'. Speed of drop indicates severity.
        *   `Vibration`: Initially 'L' or 'M'. Will only increase to 'H' *much later* as a secondary effect if bearings get damaged due to lack of oil.
        *   `EGT`: Initially 'M'. Might rise to 'H' *much later* due to secondary friction/damage.
        *   `N2`: Initially 'M'. Might fluctuate or drop if engine starts to seize (very late stage).
    *   *Detection:* Model looks for sustained 'L' in Oil Pressure.

*   **b) Bearing Wear (Generic Core Bearing)**
    *   *In-depth:* Degradation of a critical bearing supporting the main rotating shafts (N1 or N2). Caused by fatigue over time, manufacturing defect, lubrication issues, or contamination. Starts as micro-damage, increases friction and heat locally, generates vibration, sheds debris into oil (detectable by chip detectors - not in our sensor list). If ignored, can lead to bearing collapse, shaft imbalance, blade contact, engine seizure.
    *   *Sensor Signature (Simplified Scope):*
        *   `Vibration`: **Primary.** Increases from 'L'/'M' -> 'H'. May also show increased variance/spikiness.
        *   `EGT`: **Secondary.** Can increase slightly from 'M' -> 'H' due to friction/inefficiency.
        *   `N2`: **Secondary/Subtle.** Might require slightly more fuel flow to maintain cruise N2, or show slight drop.
        *   `Oil Pressure`: Assumed 'M'. No direct impact modeled here.
    *   *Detection:* Model looks for sustained 'H' in Vibration, possibly corroborated by a slight increase in EGT.

**5. How They Interact & PGM Roles (Simplified View)**

1.  **MRF (Vib1, Vib2):** Takes raw Vib1, Vib2 readings. If Vib1 reads 'H' but Vib2 reads 'M', the MRF encourages consistency, potentially outputting smoothed values closer to 'M' or 'H' for both, reducing the impact of outlier noise on one sensor before further analysis.
2.  **HMM (Vibration, EGT):** Takes discretized (potentially smoothed for Vib) readings. Tracks sensor health over time. If Vibration sensor readings become consistently 'H' *even when other sensors suggest normal operation*, the HMM might flag the *sensor* as 'Degraded' or 'Faulty'. This prevents the DBN from overreacting to a bad sensor. Similarly for EGT drift.
3.  **DBN (Core Logic):**
    *   Receives discretized EGT, N2, Oil Pressure, and Vibration readings.
    *   Uses the HMM output to *trust* the Vibration and EGT readings appropriately (e.g., down-weights evidence from a 'Faulty' sensor).
    *   Uses its CPTs (learned/defined probabilities) to reason:
        *   "If Oil Pressure evidence is 'L' (and sensor is trusted), the probability of `Lubrication_System_Health` = 'Fail' increases significantly."
        *   "If Vibration evidence is 'H' (and sensor is trusted), probability of `Engine_Core_Health` = 'Warn'/'Fail' increases."
        *   "If Vibration is 'H' AND EGT is slightly higher than usual ('H'), confidence in `Engine_Core_Health` = 'Warn'/'Fail' increases further."
        *   "If `Lubrication_System_Health` was 'Fail' at time t-1, it's likely still 'Fail' at time t unless strong contradictory evidence appears."
    *   Calculates the probabilities of component health states and overall engine health based on this integrated, time-aware, reliability-aware reasoning.


