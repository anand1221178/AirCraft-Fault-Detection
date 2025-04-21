# dbn_model.py

from pgmpy.models import DynamicBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os


# In dbn_model.py

def define_dbn_structure():
    """
    Defines the DBN structure including sensor health influencing observations.
    """
    dbn_structure = [
        # --- Intra-slice edges (time 0) ---
        (('Engine_Core_Health', 0), ('EGT_C_Discrete', 0)),
        (('Engine_Core_Health', 0), ('N2_PctRPM_Discrete', 0)),
        (('Engine_Core_Health', 0), ('Vib1_IPS_Discrete', 0)),
        (('Engine_Core_Health', 0), ('Vib2_IPS_Discrete', 0)),
        (('Lubrication_System_Health', 0), ('OilPressure_PSI_Discrete', 0)),
        # ADDED: Sensor health influences sensors
        (('EGT_Sensor_Health', 0), ('EGT_C_Discrete', 0)),
        (('Vibration_Sensor_Health', 0), ('Vib1_IPS_Discrete', 0)),
        (('Vibration_Sensor_Health', 0), ('Vib2_IPS_Discrete', 0)),

        # --- Inter-slice edges (time 0 to time 1) ---
        (('Engine_Core_Health', 0), ('Engine_Core_Health', 1)),
        (('Lubrication_System_Health', 0), ('Lubrication_System_Health', 1)),
        (('EGT_Sensor_Health', 0), ('EGT_Sensor_Health', 1)), # Sensor health persists
        (('Vibration_Sensor_Health', 0), ('Vibration_Sensor_Health', 1)), # Sensor health persists

        # Sensor readings at t=1 depend on states at t=1
        (('Engine_Core_Health', 1), ('EGT_C_Discrete', 1)),
        (('Engine_Core_Health', 1), ('N2_PctRPM_Discrete', 1)),
        (('Engine_Core_Health', 1), ('Vib1_IPS_Discrete', 1)),
        (('Engine_Core_Health', 1), ('Vib2_IPS_Discrete', 1)),
        (('Lubrication_System_Health', 1), ('OilPressure_PSI_Discrete', 1)),
         # ADDED: Sensor health influences sensors @ t=1
        (('EGT_Sensor_Health', 1), ('EGT_C_Discrete', 1)),
        (('Vibration_Sensor_Health', 1), ('Vib1_IPS_Discrete', 1)),
        (('Vibration_Sensor_Health', 1), ('Vib2_IPS_Discrete', 1)),
    ]
    dbn = DynamicBayesianNetwork(dbn_structure)
    print("--- Nodes created by DBN init ---") # Keep for debugging
    print(sorted(dbn.nodes()))
    print("-------------------------------")
    return dbn

def define_initial_cpts():
    """
    Defines CPTs for the DBN, including sensor health influence.
    Values are explicitly defined for clarity and to avoid type errors.
    State Ordering:
    - CoreHealth: 0=OK, 1=Warn, 2=Fail
    - LubHealth: 0=OK, 1=Fail
    - SensorHealth (EGT/Vib): 0=OK, 1=Degraded, 2=Failed
    - Observations (EGT,N2,OilP,Vib1,Vib2): 0=Low, 1=Medium, 2=High
    """
    cpt_list = []

    # --- Initial State Probabilities (Time 0) ---
    cpd_core_health_0 = TabularCPD(variable=('Engine_Core_Health', 0), variable_card=3, values=[[0.95], [0.04], [0.01]])
    cpt_list.append(cpd_core_health_0)
    cpd_lub_health_0 = TabularCPD(variable=('Lubrication_System_Health', 0), variable_card=2, values=[[0.98], [0.02]])
    cpt_list.append(cpd_lub_health_0)
    cpd_egt_sh_0 = TabularCPD(variable=('EGT_Sensor_Health', 0), variable_card=3, values=[[0.97], [0.02], [0.01]])
    cpt_list.append(cpd_egt_sh_0)
    cpd_vib_sh_0 = TabularCPD(variable=('Vibration_Sensor_Health', 0), variable_card=3, values=[[0.97], [0.02], [0.01]])
    cpt_list.append(cpd_vib_sh_0)

    # --- Observation CPTs (Time 0) ---

    # P(EGT_C_Discrete | Engine_Core_Health, EGT_Sensor_Health)
    egt_obs_values = [ # Explicitly defining the 2D list
        [0.10, 0.25, 0.33,  0.10, 0.25, 0.33,  0.10, 0.20, 0.33], # EGT = Low
        [0.80, 0.50, 0.34,  0.50, 0.50, 0.34,  0.20, 0.40, 0.34], # EGT = Medium
        [0.10, 0.25, 0.33,  0.40, 0.25, 0.33,  0.70, 0.40, 0.33]  # EGT = High
    ]
    cpd_egt_obs = TabularCPD(
        variable=('EGT_C_Discrete', 0), variable_card=3,
        values=egt_obs_values,
        evidence=[('Engine_Core_Health', 0), ('EGT_Sensor_Health', 0)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_egt_obs)

    # P(N2_PctRPM_Discrete | Engine_Core_Health)
    n2_obs_values = [[0.10, 0.10, 0.15], 
                     [0.80, 0.80, 0.75], 
                     [0.10, 0.10, 0.10]]
    cpd_n2_obs = TabularCPD(
        variable=('N2_PctRPM_Discrete', 0), variable_card=3,
        values=n2_obs_values,
        evidence=[('Engine_Core_Health', 0)], evidence_card=[3]
    )
    cpt_list.append(cpd_n2_obs)

    # P(OilPressure_PSI_Discrete | Lubrication_System_Health)
    oilp_obs_values = [[0.10, 0.80], 
                       [0.80, 0.10], 
                       [0.10, 0.10]]
    cpd_oilp_obs = TabularCPD(
        variable=('OilPressure_PSI_Discrete', 0), variable_card=3,
        values=oilp_obs_values,
        evidence=[('Lubrication_System_Health', 0)], evidence_card=[2]
    )
    cpt_list.append(cpd_oilp_obs)

    # P(Vib1_IPS_Discrete | Engine_Core_Health, Vibration_Sensor_Health)
    vib_obs_values = [ # Define once, reuse the list variable
        [0.60, 0.30, 0.33,  0.10, 0.30, 0.33,  0.10, 0.20, 0.33], # Vib = Low
        [0.30, 0.40, 0.34,  0.60, 0.40, 0.34,  0.25, 0.40, 0.34], # Vib = Medium
        [0.10, 0.30, 0.33,  0.30, 0.30, 0.33,  0.65, 0.40, 0.33]  # Vib = High
    ]
    cpd_vib1_obs = TabularCPD(
        variable=('Vib1_IPS_Discrete', 0), variable_card=3,
        values=vib_obs_values,
        evidence=[('Engine_Core_Health', 0), ('Vibration_Sensor_Health', 0)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_vib1_obs)

    # P(Vib2_IPS_Discrete | Engine_Core_Health, Vibration_Sensor_Health) - Use the same list
    cpd_vib2_obs = TabularCPD(
        variable=('Vib2_IPS_Discrete', 0), variable_card=3,
        values=vib_obs_values, # Reuse the Python list variable
        evidence=[('Engine_Core_Health', 0), ('Vibration_Sensor_Health', 0)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_vib2_obs)


    # --- Temporal Transition Probabilities (Slice 0 -> Slice 1) ---
    cpd_core_health_t1 = TabularCPD(variable=('Engine_Core_Health', 1), variable_card=3, values=[[0.90 , 0.10, 0.05], [0.05, 0.85, 0.21], [0.05, 0.05, 0.74]], evidence=[('Engine_Core_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_core_health_t1)
    cpd_lub_health_t1 = TabularCPD(variable=('Lubrication_System_Health', 1), variable_card=2, values=[[0.95, 0.05], [0.05, 0.95]], evidence=[('Lubrication_System_Health', 0)], evidence_card=[2])
    cpt_list.append(cpd_lub_health_t1)
    cpd_egt_sh_t1 = TabularCPD(variable=('EGT_Sensor_Health', 1), variable_card=3, values=[[0.97, 0.10, 0.01], [0.02, 0.85, 0.10], [0.01, 0.05, 0.89]], evidence=[('EGT_Sensor_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_egt_sh_t1)
    cpd_vib_sh_t1 = TabularCPD(variable=('Vibration_Sensor_Health', 1), variable_card=3, values=[[0.97, 0.10, 0.01], [0.02, 0.85, 0.10], [0.01, 0.05, 0.89]], evidence=[('Vibration_Sensor_Health', 0)], evidence_card=[3])
    cpt_list.append(cpd_vib_sh_t1)


    # --- CPTs for Observation Nodes at t=1 ---
    # Explicitly define values, reusing the lists defined above for time 0

    # P(EGT_C_Discrete(1) | Engine_Core_Health(1), EGT_Sensor_Health(1))
    cpd_egt_obs_t1 = TabularCPD(
        variable=('EGT_C_Discrete', 1), variable_card=3,
        values=egt_obs_values, # Reuse list variable
        evidence=[('Engine_Core_Health', 1), ('EGT_Sensor_Health', 1)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_egt_obs_t1)

    # P(N2_PctRPM_Discrete(1) | Engine_Core_Health(1))
    cpd_n2_obs_t1 = TabularCPD(
        variable=('N2_PctRPM_Discrete', 1), variable_card=3,
        values=n2_obs_values, # Reuse list variable
        evidence=[('Engine_Core_Health', 1)], evidence_card=[3]
    )
    cpt_list.append(cpd_n2_obs_t1)

    # P(OilPressure_PSI_Discrete(1) | Lubrication_System_Health(1))
    cpd_oilp_obs_t1 = TabularCPD(
        variable=('OilPressure_PSI_Discrete', 1), variable_card=3,
        values=oilp_obs_values, # Reuse list variable
        evidence=[('Lubrication_System_Health', 1)], evidence_card=[2]
    )
    cpt_list.append(cpd_oilp_obs_t1)

    # P(Vib1_IPS_Discrete(1) | Engine_Core_Health(1), Vibration_Sensor_Health(1))
    cpd_vib1_obs_t1 = TabularCPD(
        variable=('Vib1_IPS_Discrete', 1), variable_card=3,
        values=vib_obs_values, # Reuse list variable
        evidence=[('Engine_Core_Health', 1), ('Vibration_Sensor_Health', 1)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_vib1_obs_t1)

    # P(Vib2_IPS_Discrete(1) | Engine_Core_Health(1), Vibration_Sensor_Health(1))
    cpd_vib2_obs_t1 = TabularCPD(
        variable=('Vib2_IPS_Discrete', 1), variable_card=3,
        values=vib_obs_values, # Reuse list variable
        evidence=[('Engine_Core_Health', 1), ('Vibration_Sensor_Health', 1)],
        evidence_card=[3, 3]
    )
    cpt_list.append(cpd_vib2_obs_t1)

    # --- Final Check ---
    print(f"Defined {len(cpt_list)} CPTs for the integrated model.")
    return cpt_list

def visualize_dbn(dbn):
    """
    Visualizes the DBN template structure (connections from t=0 to t=1)
    using Graphviz for a clearer layout if available.

    """
    print("Attempting DBN visualization...")
    try:
        G_layout = nx.DiGraph(dbn.edges())
        pos = nx.drawing.nx_agraph.graphviz_layout(G_layout, prog='dot')
        print("Using Graphviz 'dot' layout.")
        graphviz_available = True
    except ImportError:
        print("PyGraphviz not found. Falling back to basic layout.")
        print("Install Graphviz and pygraphviz for better visualization:")
        print("  Ubuntu: sudo apt install graphviz; pip install pygraphviz")
        print("  macOS: brew install graphviz; pip install pygraphviz")
        print("  Conda: conda install pygraphviz python-graphviz")
        pos = nx.multipartite_layout(dbn, subset_key=1) # Basic fallback
        graphviz_available = False
    except Exception as e:
        print(f"Graphviz layout failed: {e}. Falling back to basic layout.")
        pos = nx.multipartite_layout(dbn, subset_key=1) # Basic fallback
        graphviz_available = False

    plt.figure(figsize=(18, 14)) # Adjusted size for better spacing

    nodes_t0 = [n for n in dbn.nodes() if n[1] == 0]
    nodes_t1 = [n for n in dbn.nodes() if n[1] == 1]

    # Define node types for coloring (optional)
    hidden_nodes = [n for n in dbn.nodes() if 'Health' in n[0]]
    observed_nodes = [n for n in dbn.nodes() if n not in hidden_nodes]

    # Draw nodes with different colors maybe
    nx.draw_networkx_nodes(dbn, pos, nodelist=[n for n in hidden_nodes if n[1]==0], node_color='lightcoral', node_size=4500, label='Hidden t=0')
    nx.draw_networkx_nodes(dbn, pos, nodelist=[n for n in observed_nodes if n[1]==0], node_color='lightblue', node_size=4500, label='Observed t=0')
    nx.draw_networkx_nodes(dbn, pos, nodelist=[n for n in hidden_nodes if n[1]==1], node_color='salmon', node_size=4500, label='Hidden t=1')
    nx.draw_networkx_nodes(dbn, pos, nodelist=[n for n in observed_nodes if n[1]==1], node_color='lightgreen', node_size=4500, label='Observed t=1')

    # Draw edges with increased thickness and different styles
    edges = dbn.edges()
    intra_slice_edges = [e for e in edges if e[0][1] == e[1][1]] # Edges within any slice
    inter_slice_edges = [e for e in edges if e[0][1] == 0 and e[1][1] == 1] # Edges t=0 -> t=1

    edge_width = 2.0 # Thicker lines
    arrow_size = 25 # Larger arrows

    # --- REMOVED connectionstyle from these calls ---
    nx.draw_networkx_edges(dbn, pos, edgelist=intra_slice_edges, edge_color='gray', style='dashed',
                           width=edge_width, arrows=True, arrowsize=arrow_size)
    nx.draw_networkx_edges(dbn, pos, edgelist=inter_slice_edges, edge_color='red', style='solid',
                           width=edge_width, arrows=True, arrowsize=arrow_size)
    # --- End of Change ---

    # Draw labels (
    labels = {node: f"{node[0].replace('_Discrete','').replace('_IPS','').replace('_PctRPM','').replace('_PSI','').replace('_C','')}\n(t={node[1]})" for node in dbn.nodes()}
    nx.draw_networkx_labels(dbn, pos, labels=labels, font_size=9, font_weight='bold')

    plt.title("Dynamic Bayesian Network Structure (Template: t=0 to t=1)", fontsize=18)
    # Annotate edge types if graphviz wasn't used 
    if not graphviz_available:
        plt.text(0.5, -0.05, 'Dashed: Intra-slice | Solid Red: Inter-slice (t=0 -> t=1)',
                 transform=plt.gca().transAxes, ha='center', fontsize=10)
    plt.axis('off')
    plt.tight_layout()

    # Ensure 'Data' directory exists
    data_dir = 'Data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    save_path = os.path.join(data_dir, 'dbn_structure_improved.png')
    plt.savefig(save_path, dpi=150)
    # plt.show() # Optionally display plot immediately
    plt.close() # Close the figure
    print(f"Improved DBN Structure Diagram Saved to {save_path}")

# --- (Keep check_dbn as it was) ---
def check_dbn(dbn):
    """Checks the validity of the DBN model structure and CPTs."""
    try:
        dbn.check_model()
        print("DBN Model Check Passed: Structure and CPTs are valid.")
        return True
    except Exception as e:
        print(f"DBN Model Check Failed with Error: {e}")
        return False

# --- (Keep if __name__ == "__main__" as it was) ---
if __name__ == "__main__":
    print("--- Defining DBN Structure ---")
    dbn = define_dbn_structure()
    print("DBN Structure Defined.")

    print("\n--- Defining Initial CPTs ---")
    cpt_list = define_initial_cpts()
    print(f"Defined {len(cpt_list)} CPTs.")

    print("\n--- Adding CPTs to DBN ---")
    all_added = True
    for i, cpt in enumerate(cpt_list):
        # print(f"Adding CPT {i+1}/{len(cpt_list)} for variable {cpt.variable}") # Debug print
        try:
            dbn.add_cpds(cpt)
        except Exception as e:
            print(f"\nError adding CPT for variable {cpt.variable}: {e}")
            print("CPT details:")
            print(cpt)
            all_added = False
            break

    if all_added:
        print("All CPTs added successfully.")

        print("\n--- Visualizing DBN Structure ---")
        visualize_dbn(dbn) # Call the improved visualization

        print("\n--- Checking DBN Model ---")
        check_dbn(dbn)
    else:
        print("\n--- Skipping Visualization and Model Check due to CPT errors ---")
