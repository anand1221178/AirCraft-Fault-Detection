During Prediction/Inference (when the model is running on new data):

INPUTS: The model only uses the sensor readings over time as input:

EGT_C

N2_PctRPM

OilPressure_PSI

Vib1_IPS

Vib2_IPS

(Note: These inputs will likely be processed first - Vib1/Vib2 potentially smoothed by the MRF, and then all will be discretized before feeding into the DBN/HMM)

OUTPUTS (Predictions): Based only on those sensor inputs, the model predicts the probabilities of the hidden states:

Engine_Core_Health (e.g., 'OK', 'Warn', 'Fail')

Lubrication_System_Health (e.g., 'OK', 'Fail')

Fault_Type (e.g., 'OilLeak', 'BearingWear', 'None' - often derived from the health states)

EGT_Sensor_Health (e.g., 'OK', 'Degraded', 'Failed' - from the EGT HMM)

Vibration_Sensor_Health (e.g., 'OK', 'Degraded', 'Failed' - from the Vibration HMM)

Role of Ground Truth (during Evaluation/Testing):

The columns Engine_Fault_State, EGT_Sensor_Health, and Vibration_Sensor_Health from your CSV are NOT inputs to the model when it's making predictions.

Instead, they are the "correct answers" (ground truth). You will compare the model's predicted outputs (listed above) against these ground truth labels to see how accurate your model is.

How Predictions Map to Scenarios:

The model doesn't explicitly predict the "Scenario" name (like 'BearingWear' or 'EGTSensorFail'). Instead, it predicts the underlying component health and sensor health states. We interpret these predicted states to understand which scenario the model thinks is happening:

Example 1: Model Predicts: High P(Engine_Core_Health='Fail'), High P(Fault_Type='BearingWear'), High P(EGT_Sensor_Health='OK'), High P(Vibration_Sensor_Health='OK').

Interpretation: The model believes the core is failing due to bearing wear, and the sensors are reliable. This corresponds to your 'BearingWear' scenario.

Example 2: Model Predicts: High P(Engine_Core_Health='OK'), High P(Fault_Type='None'), High P(EGT_Sensor_Health='Failed'), High P(Vibration_Sensor_Health='OK').

Interpretation: The model believes the engine core is fine, no fault is active, but the EGT sensor has failed. This corresponds to your 'EGTSensorFail' scenario.

In short: The model uses sensor readings to predict engine health, fault types, and sensor health states. You then use the ground truth labels (Engine_Fault_State, etc.) to check if those predictions were correct for the different scenarios you simulated.Almost, but let's refine that slightly for complete accuracy!

Your model will use the readings from these sensors as INPUT:

EGT_C (Exhaust Gas Temperature)

N2_PctRPM (High-Pressure Spool Speed)

OilPressure_PSI (Oil Pressure)

Vib1_IPS (Vibration Sensor 1)

Vib2_IPS (Vibration Sensor 2)

These are the OBSERVED variables. Think of these as the "symptoms" the engine is showing. Your model will look at these sensor readings at each time step to try and figure out what's going on internally within the engine.

Your model will PREDICT (as probabilities) these things as OUTPUT:

Probabilities of Engine Fault Types:

P(Fault_Type = 'OilLeak') - How likely is it that an Oil Leak is currently happening?

P(Fault_Type = 'BearingWear') - How likely is it that Bearing Wear is currently happening?

P(Fault_Type = 'None') - How likely is it that neither Oil Leak nor Bearing Wear is happening (i.e., engine is operating normally from a fault perspective)?

Probabilities of Engine Component Health States:

P(Engine_Core_Health = 'OK') / 'Warn' / 'Fail' - What is the health status of the engine's core rotating components (compressor, turbine, bearings)? (Remember, you're likely going to discretize 'Engine_Core_Health' into these states).

P(Lubrication_System_Health = 'OK') / 'Fail' - What is the health status of the lubrication system? (Likely discretized into 'OK' and 'Fail').

Probabilities of Sensor Health States (Specifically for HMM-monitored sensors):

P(EGT_Sensor_Health = 'OK') / 'Degraded' / 'Failed' - What is the health of the EGT sensor itself? (Predicted by the EGT HMM).

P(Vibration_Sensor_Health = 'OK') / 'Degraded' / 'Failed' - What is the health of the Vibration sensor system? (Predicted by the Vibration HMM).

Crucially, the Engine_Fault_State, EGT_Sensor_Health, and Vibration_Sensor_Health columns in your CSV are NOT used as inputs to your model during prediction.

Why? Because these are the GROUND TRUTH LABELS. They are the answers we know are correct in our simulated data. We use these ground truth labels to:

Train the model: The model learns to associate patterns in the sensor readings (inputs) with the corresponding ground truth engine/sensor states.

Evaluate the model: After training, we test the model on new simulated data and compare its predictions (outputs) to the known ground truth to see how accurate it is.

Think of it like this:

Imagine a doctor diagnosing a patient.

Inputs (Symptoms): Temperature (like EGT), Heart Rate (like N2), Blood Pressure (like Oil Pressure), maybe observations of shaking/tremors (like Vibration).

Output (Diagnosis): Probability of having Flu, Probability of having Pneumonia, Probability of having a Common Cold, etc., and perhaps also a general health assessment (Good, Fair, Poor). The doctor also might assess the reliability of their thermometer if it's giving odd readings (like your sensor health HMMs).

Ground Truth (What the patient actually has): In real life, you only find this out definitively later (maybe after tests). In your simulation, you pre-define this (e.g., you decide to simulate a "BearingWear" scenario, so you know the ground truth is Bearing Wear).

So, your model is learning to be a diagnostic "doctor" for your simulated engine, using only the "symptoms" (sensor readings) to infer the hidden "diseases" (faults) and "overall health" of the engine components and sensors.