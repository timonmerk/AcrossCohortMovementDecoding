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
    model=xgboost.XGBClassifier(),
    ML_model_name="XGB",
)

# r correlation df

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

ap_runner.cross_val_p2p_RMAP(df_rmap=df_rmap_corr)
