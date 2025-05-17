# In DBN/update_cpd.py

from pgmpy.factors.discrete import TabularCPD
import numpy as np
from dbn_inference import infer_marginals_dataframe, prepare_evidence_sequence
from config import HEALTH_NODE

def dirichlet_posterior(old_cpd: TabularCPD, 
                        child_state_idx: int,
                        parent_config_col_idx: int,
                        alpha: float = 0.1):
    # print(f"-> ENTERING dirichlet_posterior for {old_cpd.variable} with child_idx={child_state_idx}, parent_col_idx={parent_config_col_idx}, alpha={alpha}")
    # ... (alpha check, boundary checks for indices) ...
    # if alpha <= 0: alpha = 1e-6
    # if not (0 <= child_state_idx < old_cpd.values.shape[0]): return old_cpd
    # if not (0 <= parent_config_col_idx < old_cpd.values.shape[1]): return old_cpd

    original_column_probs = old_cpd.values[:, parent_config_col_idx].copy() # For comparison

    pseudo_counts = old_cpd.values.copy() * (1.0 / alpha)
    pseudo_counts[child_state_idx, parent_config_col_idx] += 1.0
    new_normalized_values = pseudo_counts / pseudo_counts.sum(axis=0, keepdims=True)

    # # --- DEBUG: Check if values actually changed ---
    # # Pick the specific column that was updated
    # updated_column_probs = new_normalized_values[:, parent_config_col_idx]
    # if not np.allclose(original_column_probs, updated_column_probs):
    #     print(f"  DEBUG (dirichlet_posterior) for CPD {old_cpd.variable}, child_idx={child_state_idx}, parent_col_idx={parent_config_col_idx}:")
    #     print(f"    Old probs for this column: {original_column_probs}")
    #     print(f"    New probs for this column: {updated_column_probs}")
    #     print(f"    Difference: {updated_column_probs - original_column_probs}")
    #     # else:
    #     #     # This might happen if alpha is huge and old_cpd.values are tiny, or if update is to same state
    #     print(f"  DEBUG (dirichlet_posterior) for CPD {old_cpd.variable}: No significant change in probabilities for column {parent_config_col_idx}.")
    # # --- END DEBUG ---


    # ... (rest of the function to reconstruct and return TabularCPD with new_normalized_values) ...
    evidence_nodes = old_cpd.get_evidence()
    evidence_cards_list = None
    if evidence_nodes:
        evidence_cards_list = []
        for ev_node in evidence_nodes: 
            parent_idx_in_scope = -1
            for i, node_in_scope in enumerate(old_cpd.variables):
                if node_in_scope == ev_node: 
                    parent_idx_in_scope = i
                    break
            if parent_idx_in_scope != -1:
                evidence_cards_list.append(old_cpd.cardinality[parent_idx_in_scope])
            else:
                raise ValueError(f"PGMPY INCONSISTENCY: Parent {ev_node} not in CPD scope variables {old_cpd.variables}")

    return TabularCPD(
        variable=old_cpd.variable,
        variable_card=old_cpd.variable_card,
        values=new_normalized_values, # Use the updated values
        evidence=evidence_nodes, 
        evidence_card=evidence_cards_list,
        state_names=old_cpd.state_names 
    )


# In DBN/update_cpd.py

# Make sure dirichlet_posterior is defined above this function in the same file
# and that prepare_evidence_sequence and infer_marginals_dataframe are imported at the top.

def update_cpds_online(dbn, unit_df, health_node_name=None, obs_nodes=None, alpha=0.1):
    evidence_seq_dict_list = prepare_evidence_sequence(unit_df) 
    update_attempts = 0
    successful_updates = 0

    print(f"DEBUG (update_cpds_online): Starting online updates for unit. Alpha={alpha}. Evidence length: {len(evidence_seq_dict_list)}")

    for t, ev_t_slice0_dict in enumerate(evidence_seq_dict_list[:-1]):
        ev_t_plus_1_slice0_dict = evidence_seq_dict_list[t+1]
        # print(f"  Time step t={t}") # Optional: print time step

        for cpd_to_update in dbn.get_cpds(time_slice=1):
            # print(f"    Processing CPD: {cpd_to_update.variable}") # Optional: print CPD being processed
            child_node_slice1 = cpd_to_update.variable 
            parent_nodes = cpd_to_update.get_evidence() 
            parent_config_column_index = 0 

            child_base_name = child_node_slice1[0]
            child_key_in_evidence = f"{child_base_name}_0"

            if child_key_in_evidence not in ev_t_plus_1_slice0_dict:
                print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Child '{child_key_in_evidence}' not in t+1 evidence dict. Keys: {ev_t_plus_1_slice0_dict.keys()}")
                continue
            
            observed_child_state_val = ev_t_plus_1_slice0_dict[child_key_in_evidence]
            observed_child_state_index = -1

            try: 
                cpd_state_names = getattr(cpd_to_update, 'state_names', {})
                child_node_state_names_list = cpd_state_names.get(child_node_slice1)

                if child_node_state_names_list is not None:
                    if isinstance(child_node_state_names_list, dict) and observed_child_state_val in child_node_state_names_list:
                        observed_child_state_index = child_node_state_names_list[observed_child_state_val]
                    elif isinstance(child_node_state_names_list, list) and observed_child_state_val in child_node_state_names_list:
                        observed_child_state_index = child_node_state_names_list.index(observed_child_state_val)
                    elif not (isinstance(child_node_state_names_list, (dict,list)) and observed_child_state_val in child_node_state_names_list):
                         observed_child_state_index = int(observed_child_state_val)
                elif hasattr(cpd_to_update, 'get_state_no') and child_node_slice1 in cpd_to_update.get_variables(): 
                     observed_child_state_index = cpd_to_update.get_state_no(child_node_slice1, observed_child_state_val)
                else: 
                    observed_child_state_index = int(observed_child_state_val) 
                
                if not (0 <= observed_child_state_index < cpd_to_update.variable_card):
                    print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Child state index {observed_child_state_index} (from value {observed_child_state_val}) out of bounds for card {cpd_to_update.variable_card}.")
                    continue 
            except (ValueError, TypeError, KeyError, AttributeError) as e_child_idx: 
                print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Error converting child state '{observed_child_state_val}' to index: {e_child_idx}")
                continue
            
            if parent_nodes:
                all_parents_found_and_valid = True
                parent_state_indices_for_col_calc = []

                for p_idx, p_node in enumerate(parent_nodes): 
                    p_base_name = p_node[0]
                    p_key_in_evidence = f"{p_base_name}_0"
                    current_evidence_source_dict = ev_t_slice0_dict if p_node[1] == 0 else ev_t_plus_1_slice0_dict
                    
                    if p_key_in_evidence not in current_evidence_source_dict:
                        print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Parent '{p_key_in_evidence}' (for {p_node}) not in its evidence dict. Keys: {current_evidence_source_dict.keys()}")
                        all_parents_found_and_valid = False; break
                    
                    parent_state_val = current_evidence_source_dict[p_key_in_evidence]
                    parent_state_idx = -1
                    try: 
                        parent_card_from_cpd = -1
                        parent_idx_in_cpd_vars = -1
                        for i_var, node_in_cpd_scope in enumerate(cpd_to_update.variables):
                            if node_in_cpd_scope == p_node: parent_idx_in_cpd_vars = i_var; break
                        if parent_idx_in_cpd_vars != -1:
                            parent_card_from_cpd = cpd_to_update.cardinality[parent_idx_in_cpd_vars]
                        else: 
                            print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Parent node {p_node} not found in CPD variables {cpd_to_update.variables} for cardinality.")
                            all_parents_found_and_valid = False; break

                        parent_node_state_names_list_p = cpd_state_names.get(p_node)
                        if parent_node_state_names_list_p is not None:
                            if isinstance(parent_node_state_names_list_p, dict) and parent_state_val in parent_node_state_names_list_p:
                                parent_state_idx = parent_node_state_names_list_p[parent_state_val]
                            elif isinstance(parent_node_state_names_list_p, list) and parent_state_val in parent_node_state_names_list_p:
                                parent_state_idx = parent_node_state_names_list_p.index(parent_state_val)
                            elif not (isinstance(parent_node_state_names_list_p, (dict,list)) and parent_state_val in parent_node_state_names_list_p):
                                 parent_state_idx = int(parent_state_val)
                        elif hasattr(cpd_to_update, 'get_state_no') and p_node in cpd_to_update.get_variables():
                             parent_state_idx = cpd_to_update.get_state_no(p_node, parent_state_val)
                        else:
                            parent_state_idx = int(parent_state_val)
                        
                        if not (0 <= parent_state_idx < parent_card_from_cpd): 
                            print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Parent {p_node} state index {parent_state_idx} (from value {parent_state_val}) out of bounds for card {parent_card_from_cpd}.")
                            all_parents_found_and_valid = False; break
                        parent_state_indices_for_col_calc.append(parent_state_idx)
                    except Exception as e_parent_idx: 
                        print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Error converting parent state '{parent_state_val}' for {p_node} to index: {e_parent_idx}")
                        all_parents_found_and_valid = False; break
                
                if not all_parents_found_and_valid:
                    continue

                multiplier = 1
                current_col_idx_val = 0 
                for i in range(len(parent_nodes) - 1, -1, -1):
                    p_node_for_col = parent_nodes[i]
                    p_state_idx_for_col = parent_state_indices_for_col_calc[i]
                    
                    parent_card_for_col = -1; idx_in_vars = -1
                    for i_v, var_in_s in enumerate(cpd_to_update.variables):
                        if var_in_s == p_node_for_col: idx_in_vars = i_v; break
                    if idx_in_vars != -1: parent_card_for_col = cpd_to_update.cardinality[idx_in_vars]
                    else: 
                        print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Parent {p_node_for_col} for column index calc not found in scope {cpd_to_update.variables}")
                        current_col_idx_val = -1; break
                    if parent_card_for_col <= 0: # Card must be positive
                        print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Invalid parent_card_for_col ({parent_card_for_col}) for {p_node_for_col}.")
                        current_col_idx_val = -1; break
                        
                    current_col_idx_val += p_state_idx_for_col * multiplier
                    multiplier *= parent_card_for_col
                
                if current_col_idx_val == -1: continue 
                parent_config_column_index = current_col_idx_val 

            if not (0 <= observed_child_state_index < cpd_to_update.values.shape[0] and \
                    0 <= parent_config_column_index < cpd_to_update.values.shape[1]):
                print(f"    SKIP (t={t}, CPD={cpd_to_update.variable}): Final indices for dirichlet_posterior out of bounds. ChildIdx: {observed_child_state_index} (max: {cpd_to_update.values.shape[0]-1}), ParentColIdx: {parent_config_column_index} (max: {cpd_to_update.values.shape[1]-1}).")
                continue
            
            update_attempts +=1
            # print(f"DEBUG (update_cpds_online): Attempting to update CPD for {cpd_to_update.variable}") 
            # print(f"  Child state index to pass: {observed_child_state_index}")
            # print(f"  Parent config column index to pass: {parent_config_column_index}")
            # print(f"  Alpha to pass: {alpha}")
            # if 0 <= parent_config_column_index < cpd_to_update.values.shape[1]:
            #     if 0 <= observed_child_state_index < cpd_to_update.values.shape[0]:
            #         print(f"  Old values for target cell ({observed_child_state_index}, {parent_config_column_index}) in CPD {cpd_to_update.variable}: {cpd_to_update.values[observed_child_state_index, parent_config_column_index]}")
            #     else:
            #         print(f"  Skipping print of old values: child_state_index {observed_child_state_index} out of bounds for rows {cpd_to_update.values.shape[0]}")
            # else:
            #     print(f"  Skipping print of old values: parent_config_column_index {parent_config_column_index} out of bounds for columns {cpd_to_update.values.shape[1]}")
            
            try:
                # print(f"  Calling dirichlet_posterior for CPD {cpd_to_update.variable} with child_idx={observed_child_state_index}, parent_col_idx={parent_config_column_index}, alpha={alpha}")
                updated_cpd = dirichlet_posterior(
                    old_cpd=cpd_to_update, 
                    child_state_idx=observed_child_state_index,    
                    parent_config_col_idx=parent_config_column_index, 
                    alpha=alpha                                       
                )
                if updated_cpd is not cpd_to_update : 
                    if not np.allclose(cpd_to_update.values, updated_cpd.values):
                        successful_updates += 1
                    dbn.remove_cpds(cpd_to_update) 
                    dbn.add_cpds(updated_cpd)
            except Exception as e_update:
                print(f"    Error during dirichlet_posterior or CPD replacement for {cpd_to_update.variable}: {e_update}")

    print(f"DEBUG (update_cpds_online): Finished online updates. Total attempts: {update_attempts}, Successful value changes: {successful_updates}")
    results_df_adapted = infer_marginals_dataframe(unit_df, model=dbn)
    return results_df_adapted