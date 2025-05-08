
# Aircraft Engine Fault Detection – System Overview

## 1. The Core Problem

We aim to detect potential faults in an aircraft engine *before* they become critical, using real-time sensor data that may be noisy or inconsistent. Our system must:
- Understand how different engine components influence sensor behavior.
- Reason about component degradation over time.
- Handle unreliable signals due to noisy sensors.
- Encourage consistency across physically related sensor readings.

## 2. The Goal

To build a probabilistic diagnostic model combining a **Dynamic Bayesian Network (DBN)** and a **simple Markov Random Field (MRF)**. The system processes readings from 4 key sensors, accounts for sensor reliability (via DBN node structure), and enforces spatial consistency (via MRF smoothing), to output the probabilities of:
- Faults: `Oil Leak`, `Bearing Wear`
- Health States: `Engine_Core_Health`, `Lubrication_System_Health`

## 3. The Sensors

Each sensor is discretized into 3 levels: **Low (L)**, **Medium (M)**, **High (H)** — based on cruise conditions as the 'Medium' baseline.

### a) EGT (Exhaust Gas Temperature)
- Measures combustion heat post-turbine.
- Correlates moderately with `Bearing Wear`.
- Used as key input to `Engine_Core_Health`.
- No HMM applied; data used directly after discretization.

### b) N2 (High-Pressure Spool RPM)
- Measures engine core rotational speed.
- Weak correlation with faults, but reflects engine workload.
- Input to `Engine_Core_Health`.
- Assumed reliable.

### c) Oil Pressure
- Measures lubrication system pressure.
- **Primary indicator** of `Oil Leak` fault.
- Input to `Lubrication_System_Health`.
- Assumed reliable and directly used.

### d) Vibration (Vib1, Vib2)
- Measures engine mechanical vibration.
- **Primary indicator** of `Bearing Wear`.
- Inputs to `Engine_Core_Health`.
- MRF applied across Vib1 and Vib2 to reduce noise and enforce coherence.

## 4. The Faults

### a) Oil Leak
- Primary signal: drop in `Oil Pressure` from M → L.
- Vibration and EGT may be affected only at later stages.
- Detection: sustained low oil pressure → `Lubrication_System_Health = Fail`.

### b) Bearing Wear
- Primary signal: increase in `Vibration` from L/M → H.
- Secondary: slight rise in `EGT`.
- Detection: sustained high vibration → `Engine_Core_Health = Warn/Fail`.

## 5. Model Components

### 1. MRF (Vib1 ↔ Vib2)
- Ensures that vibration readings from both sensors are consistent before they influence DBN inference.
- Reduces the effect of random noise spikes on just one channel.

### 2. DBN (Dynamic Bayesian Network)
- Encodes relationships between component health, sensor evidence, and time-step transitions.
- Models:
  - Probabilistic evolution of `Engine_Core_Health`, `Lubrication_System_Health`
  - Sensor health indirectly via conditional influence (no explicit HMM state now)
  - Evidence integration over time, e.g.:
    - "Low oil pressure → likely lubrication failure"
    - "High vibration + high EGT → likely core damage"
- State at time `t` depends on both current sensor readings and prior state `t-1`.