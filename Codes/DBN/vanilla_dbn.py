# vanilla_dbn_model.py 
# (Simpler version of dbn_model.py)

from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

# NOTE: No visualization or check_model functions needed here usually,
# assuming the structure is simpler and less prone to definition errors.

def define_vanilla_dbn_structure():
    """
    Defines a simpler DBN structure WITHOUT sensor health nodes.
    Observations depend only on component health.
    """
    dbn_structure = [
        # --- Intra-slice edges (time 0) ---
        (('Engine_Core_Health', 0), ('EGT_C_Discrete', 0)),
        (('Engine_Core_Health', 0), ('N2_PctRPM_Discrete', 0)),
        (('Engine_Core_Health', 0), ('Vib1_IPS_Discrete', 0)), # Still use separate Vib nodes
        (('Engine_Core_Health', 0), ('Vib2_IPS_Discrete', 0)), # as evidence comes that way
        (('Lubrication_System_Health', 0), ('OilPressure_PSI_Discrete', 0)),

        # --- Inter-slice edges (time 0 to time 1) ---
        (('Engine_Core_Health', 0), ('Engine_Core_Health', 1)),
        (('Lubrication_System_Health', 0), ('Lubrication_System_Health', 1)),
        
        # Sensor readings at t=1 depend on health states at t=1
        (('Engine_Core_Health', 1), ('EGT_C_Discrete', 1)),
        (('Engine_Core_Health', 1), ('N2_PctRPM_Discrete', 1)),
        (('Engine_Core_Health', 1), ('Vib1_IPS_Discrete', 1)),
        (('Engine_Core_Health', 1), ('Vib2_IPS_Discrete', 1)),
        (('Lubrication_System_Health', 1), ('OilPressure_PSI_Discrete', 1)),
    ]
    dbn = DynamicBayesianNetwork(dbn_structure)
    return dbn

def define_vanilla_initial_cpts():
    """
    Defines CPTs for the Vanilla DBN. Observation CPTs only depend
    on component health states.
    State Ordering:
    - CoreHealth: 0=OK, 1=Warn, 2=Fail
    - LubHealth: 0=OK, 1=Fail
    - Observations: 0=Low, 1=Medium, 2=High
    """
    cpt_list = []

    # --- Initial State Probabilities (Time 0) ---
    cpd_core_health_0 = TabularCPD(variable=('Engine_Core_Health', 0), variable_card=3, values=[[0.95], [0.04], [0.01]])
    cpt_list.append(cpd_core_health_0)
    cpd_lub_health_0 = TabularCPD(variable=('Lubrication_System_Health', 0), variable_card=2, values=[[0.98], [0.02]])
    cpt_list.append(cpd_lub_health_0)

    # --- Observation CPTs (Time 0) ---

    # P(EGT_C_Discrete | Engine_Core_Health) - Simpler dependency
    egt_obs_values = [[0.10, 0.10, 0.05], [0.80, 0.50, 0.25], [0.10, 0.40, 0.70]] # EGT(L,M,H)|Core(OK,W,F)
    cpd_egt_obs = TabularCPD(
        variable=('EGT_C_Discrete', 0), variable_card=3,
        values=egt_obs_values,
        evidence=[('Engine_Core_Health', 0)], evidence_card=[3]
    )
    cpt_list.append(cpd_egt_obs)

    # P(N2_PctRPM_Discrete | Engine_Core_Health)
    n2_obs_values = [[0.10, 0.10, 0.15], [0.80, 0.80, 0.75], [0.10, 0.10, 0.10]] # N2(L,M,H)|Core(OK,W,F)
    cpd_n2_obs = TabularCPD(
        variable=('N2_PctRPM_Discrete', 0), variable_card=3,
        values=n2_obs_values,
        evidence=[('Engine_Core_Health', 0)], evidence_card=[3]
    )
    cpt_list.append(cpd_n2_obs)

    # P(OilPressure_PSI_Discrete | Lubrication_System_Health)
    oilp_obs_values = [[0.10, 0.80], [0.80, 0.15], [0.10, 0.05]] # OilP(L,M,H)|Lub(OK,F)
    cpd_oilp_obs = TabularCPD(
        variable=('OilPressure_PSI_Discrete', 0), variable_card=3,
        values=oilp_obs_values,
        evidence=[('Lubrication_System_Health', 0)], evidence_card=[2]
    )
    cpt_list.append(cpd_oilp_obs)

    # P(Vib1_IPS_Discrete | Engine_Core_Health) - Simpler dependency
    vib_obs_values = [[0.60, 0.10, 0.05], [0.35, 0.60, 0.25], [0.05, 0.30, 0.70]] # Vib(L,M,H)|Core(OK,W,F)
    cpd_vib1_obs = TabularCPD(
        variable=('Vib1_IPS_Discrete', 0), variable_card=3,
        values=vib_obs_values,
        evidence=[('Engine_Core_Health', 0)], evidence_card=[3]
    )
    cpt_list.append(cpd_vib1_obs)

    # P(Vib2_IPS_Discrete | Engine_Core_Health) - Simpler dependency
    cpd_vib2_obs = TabularCPD(
        variable=('Vib2_IPS_Discrete', 0), variable_card=3,
        values=vib_obs_values, # Reuse same values
        evidence=[('Engine_Core_Health', 0)], evidence_card=[3]
    )
    cpt_list.append(cpd_vib2_obs)

    # --- Temporal Transition Probabilities (Slice 0 -> Slice 1) ---
    cpd_core_health_t1 = TabularCPD(variable=('Engine_Core_Health', 1), variable_card=3, values=[[0.95, 0.10, 0.01], [0.04, 0.85, 0.19], [0.01, 0.05, 0.80]], evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_core_health_t1)
    cpd_lub_health_t1 = TabularCPD(variable=('Lubrication_System_Health', 1), variable_card=2, values=[[0.98, 0.00], [0.02, 1.00]], evidence=[('Lubrication_System_Health', 0)], evidence_card=[2])
    cpt_list.append(cpd_lub_health_t1)

    # --- CPTs for Observation Nodes at t=1 ---
    cpd_egt_obs_t1 = TabularCPD(variable=('EGT_C_Discrete', 1), variable_card=3, values=egt_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_egt_obs_t1)
    cpd_n2_obs_t1 = TabularCPD(variable=('N2_PctRPM_Discrete', 1), variable_card=3, values=n2_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_n2_obs_t1)
    cpd_oilp_obs_t1 = TabularCPD(variable=('OilPressure_PSI_Discrete', 1), variable_card=3, values=oilp_obs_values, evidence=[('Lubrication_System_Health', 1)], evidence_card=[2])
    cpt_list.append(cpd_oilp_obs_t1)
    cpd_vib1_obs_t1 = TabularCPD(variable=('Vib1_IPS_Discrete', 1), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_vib1_obs_t1)
    cpd_vib2_obs_t1 = TabularCPD(variable=('Vib2_IPS_Discrete', 1), variable_card=3, values=vib_obs_values, evidence=[('Engine_Core_Health', 1)], evidence_card=[3])
    cpt_list.append(cpd_vib2_obs_t1)

    print(f"Defined {len(cpt_list)} CPTs for the Vanilla DBN model.") # Should be 14
    return cpt_list

# You don't necessarily need to run this file directly, 
# but import these functions into your experiment script.