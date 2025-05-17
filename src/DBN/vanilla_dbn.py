# DBN/vanilla_dbn.py  – final working version
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from config import OBSERVATION_NODES
import numpy as np


def build_vanilla_dbn() -> DBN:
    """
    Extremely simple DBN:
      • one 4-state hidden node per sensor
      • self-transition only
      • slice-0 node has *string* name (sensor_X_disc)
      • slice-1 node has tuple name (sensor_X_disc, 1)
    """
    model = DBN()

    prior      = np.full((4, 1), 0.25)   # uniform prior
    transition = np.full((4, 4), 0.25)   # uniform transition

    for sensor in OBSERVATION_NODES:
        # 1️⃣  add the plain slice-0 node
        model.add_node(sensor)

        # 2️⃣  add the tuple nodes used for dynamics
        model.add_nodes_from([(sensor, 0), (sensor, 1)])

        # 3️⃣  connect slice 0 → slice 1
        model.add_edge((sensor, 0), (sensor, 1))

        # ----------  CPDs  ----------
        #   slice-0 (string-named variable)
        cpd_t0 = TabularCPD(
            variable=sensor,            # ⇐ string
            variable_card=4,
            values=prior
        )

        #   slice-1 (tuple-named variable with evidence)
        cpd_t1 = TabularCPD(
            variable=(sensor, 1),
            variable_card=4,
            values=transition,
            evidence=[(sensor, 0)],
            evidence_card=[4]
        )

        model.add_cpds(cpd_t0, cpd_t1)

    # sanity-check: will raise if anything is missing
    model.check_model()
    return model
