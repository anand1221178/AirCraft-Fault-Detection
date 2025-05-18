# dbn_model.py  – final, minimal learner
from pgmpy.models import DynamicBayesianNetwork as DBN
import pandas as pd
from config import HEALTH_NODE, OBSERVATION_NODES, STATE_ORDER

STATE_INDEX = {s: i for i, s in enumerate(STATE_ORDER)}

def create_cmaps_dbn():
    model = DBN()
    for obs in OBSERVATION_NODES:
        model.add_edge((HEALTH_NODE, 0), (obs, 0))
        model.add_edge((HEALTH_NODE, 1), (obs, 1))
    model.add_edge((HEALTH_NODE, 0), (HEALTH_NODE, 1))
    return model

def _two_slice_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, df_u in df.groupby("unit"):
        for i in range(len(df_u) - 1):
            rec = {}
            for v in [HEALTH_NODE] + OBSERVATION_NODES:
                rec[(v, 0)] = (
                    STATE_INDEX[df_u.iloc[i][v]]
                    if v == HEALTH_NODE else int(df_u.iloc[i][v])
                )
                rec[(v, 1)] = (
                    STATE_INDEX[df_u.iloc[i + 1][v]]
                    if v == HEALTH_NODE else int(df_u.iloc[i + 1][v])
                )
            rows.append(rec)

    big = pd.DataFrame.from_records(rows)


    for col in big.columns:
        if col[0] == HEALTH_NODE:               # health: 3 states
            big[col] = pd.Categorical(big[col], categories=[0,1,2])
        else:                                   # sensor: use 0…max bin
            n_bins = int(df[col[0]].max()) + 1
            big[col] = pd.Categorical(big[col], categories=list(range(n_bins)))


    return big


def learn_cpts_from_data(model, train_df):
    data = _two_slice_df(train_df)


    model.fit(data)                 


    for cpd in model.get_cpds():
        vals = cpd.values
        if vals.ndim == 1:          # reshape root CPDs to 2-D
            vals = vals.reshape(-1, 1)

        parents = cpd.get_evidence()
        full_cols = 1
        for p in parents:
            full_cols *= len(model.get_cpds(node=p).state_names[p])

        if vals.shape[1] < full_cols:        # missing combinations ➜ pad
            pad = [[1e-3] * (full_cols - vals.shape[1])
                   for _ in range(vals.shape[0])]
            vals = np.hstack([vals, pad])
            vals = vals / vals.sum(axis=0, keepdims=True)

            # rebuild CPD with the padded matrix
            from pgmpy.factors.discrete import TabularCPD
            new_cpd = TabularCPD(
                variable       = cpd.variable,
                variable_card  = cpd.variable_card,
                values         = vals,
                evidence       = parents or None,
                evidence_card  = [len(model.get_cpds(node=p).state_names[p])
                                  for p in parents] if parents else None,
                state_names    = cpd.state_names,
            )
            model.remove_cpds(cpd)
            model.add_cpds(new_cpd)

    model.check_model()
    return model

