# File: dbn_inference.py
# Description: Inference functions for DBN model on C-MAPSS data

from pgmpy.inference import DBNInference
from config import OBSERVATION_NODES, HEALTH_NODE, STATE_ORDER
import pandas as pd
import numpy as np


def prepare_evidence_sequence(unit_df):
    """
    Prepares a list of time-indexed evidence dictionaries for DBN inference.
    Each entry corresponds to a time step and contains observed values for
    all observation nodes AND THE HEALTH_NODE at time slice 0.
    """
    evidence_seq = []
    for _, row in unit_df.iterrows():
        evidence_t = {}
        # Add observation nodes
        for obs_node_name in OBSERVATION_NODES: # obs_node_name is like "sensor_2_disc"
            if obs_node_name in row: # Check if sensor column exists in the row
                evidence_t[f"{obs_node_name}_0"] = row[obs_node_name]
            evidence_t[f"{HEALTH_NODE}_0"] = row[HEALTH_NODE]
            
        evidence_seq.append(evidence_t)
    return evidence_seq


def infer_health_states_tuples(sequence_df, model):
    """
    Inference function for Full DBN which uses tuple-based nodes (e.g., (sensor, 0)).
    """
    infer = DBNInference(model)

    evidence_seq = []
    for _, row in sequence_df.iterrows():
        evidence = {(obs, 0): row[obs] for obs in OBSERVATION_NODES}
        evidence_seq.append(evidence)

    predicted_states = []
    for t in range(len(evidence_seq)):
        # REPLACE the line you just changed with the following:
        marginals = infer.query(variables=[(HEALTH_NODE, 0)],
                                evidence=evidence_seq[t])

        state = int(marginals[(HEALTH_NODE, 0)].values.argmax())
        predicted_states.append(state)

    return predicted_states

import pandas as pd
import numpy as np

def infer_marginals_dataframe(sequence_df, model):
    """
    Runs inference over an engine’s time-series and returns a DataFrame
    with per-timestep marginals for HEALTH_NODE.

    Returns:
        pd.DataFrame with columns like:
            P(Engine_Core_Health=Healthy), P(...=Degrading), P(...=Fail)
    """
    infer = DBNInference(model)
    evidence_seq_for_df = []
    for _, row in sequence_df.iterrows():
        evidence = {}
        for obs_node_name in OBSERVATION_NODES: # obs_node_name are base names like 'sensor_2_disc'
            if obs_node_name in row:
                # Convert the integer bin value from data to a string to match CPD state_names
               evidence[(obs_node_name, 0)] = int(row[obs_node_name])
        evidence_seq_for_df.append(evidence) 

    marginals = []
    for t in range(len(evidence_seq_for_df)):
        q = infer.query(variables=[(HEALTH_NODE, 0)], evidence=evidence_seq_for_df[t])
        marg = q[(HEALTH_NODE, 0)].values
        marginals.append(marg)

    probs = np.array(marginals)  # shape = (T, len(STATE_ORDER))
    cols  = [f"P({HEALTH_NODE}={s})" for s in STATE_ORDER]
    df    = pd.DataFrame(probs, columns=cols, index=sequence_df.index)

    return df
