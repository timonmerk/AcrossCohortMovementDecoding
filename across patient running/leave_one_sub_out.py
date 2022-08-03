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
)

# 1. make ch_all dict

# 2.
PATH_ch_all = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft\channel_all.npy"


ap_runner = nm_across_patient_decoding.AcrossPatientRunner(
    outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft",
    cohorts=["Beijing", "Pittsburgh", "Berlin"],
    use_nested_cv=False,
    ML_model_name="LMUPDRS",
)

# r correlation df

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")
df_updrs = pd.read_csv("df_updrs.csv")

for val_approach in [
    "leave_1_cohort_out",
    "leave_1_sub_out_within_coh",
    "leave_1_sub_out_across_coh",
]:
    ap_runner.cross_val_approach_RMAP(
        val_approach=val_approach,
        df_rmap=df_rmap_corr,
        add_UPDRS=True,
        df_updrs=df_updrs,
    )
