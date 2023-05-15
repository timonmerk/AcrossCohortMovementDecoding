import numpy as np
import os
import pandas as pd
from sklearn import metrics, linear_model, model_selection
import xgboost

import py_neuromodulation
from py_neuromodulation import (
    nm_decode,
    nm_RMAP,
    nm_cohortwrapper,
    nm_across_patient_decoding,
    nm_plots,
)

updrs_scores_Berlin = {
    "001": None,
    "002": 21,
    "003": 31,
    "004": 36,
    "005": 31,
    "006": 36,
    "007": 59,
    "008": 24,
    "009": 27,
    "010": 30,
    "011": 39,
    "012": 31,
    "013": 35,
}

updrs_scores_Beijing = {
    "FOG006": 61,
    "FOG008": 70,
    "FOG010": 51,
    "FOG011": 52,
    "FOG012": 58,
    "FOG014": 41,
    "FOG016": 55,
    "FOG021" : 39,
    "FOG022": 27,
    "FOGC001": 47,
}

updrs_scores_Pittsburgh = {
    "000": 28,
    "001": 27,
    "002": 40,
    "003": None,
    "004": 33,
    "005": 31,
    "006": 32,
    "007": 52,
    "008": 55,
    "009": 50,
    "010": 62,
    "011": 54,
    "012": 34,
    "013": 48,
    "014": 31,
    "015": 42,
}

df = pd.DataFrame(pd.Series(updrs_scores_Pittsburgh))
df = df.reset_index()
df["cohort"] = "Pittsburgh"
df_p = df.rename(columns={"index": "sub", 0: "UPDRS_total"})

df = pd.DataFrame(pd.Series(updrs_scores_Beijing))
df = df.reset_index()
df["cohort"] = "Beijing"
df_b = df.rename(columns={"index": "sub", 0: "UPDRS_total"})

df = pd.DataFrame(pd.Series(updrs_scores_Berlin))
df = df.reset_index()
df["cohort"] = "Berlin"
df_ber = df.rename(columns={"index": "sub", 0: "UPDRS_total"})
df_updrs = pd.concat([df_b, df_p, df_ber])

# first, plot correlation with RMAP selected channels

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")
df_rmap_corr = df_rmap_corr.query("cohort != 'Washington'")
df_rmap_corr = df_rmap_corr.query("cohort != 'Washington' and cohort != 'Pittsburgh'")

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")
df_rmap_corr = df_rmap_corr.query("cohort == 'Beijing' or cohort == 'Berlin'")
df_comb = pd.merge(df_rmap_corr, df_updrs, on=["sub", "cohort"])
df_comb = df_comb.dropna()

nm_plots.reg_plot(
    data=df_comb,
    x_col="performance_test",
    y_col="UPDRS_total",
    out_path_save=os.path.join("figure", "updrs_corr_per.pdf"),
)


nm_plots.reg_plot(
    data=df_comb,
    x_col="performance_test",
    y_col="UPDRS_total",
    out_path_save=None,
)