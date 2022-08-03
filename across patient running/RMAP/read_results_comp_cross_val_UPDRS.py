import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn import metrics, linear_model, model_selection

import py_neuromodulation
from py_neuromodulation import nm_decode, nm_RMAP, nm_plots

# 1. leave one cohort out
PATH_CROSS_VAL_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"

df = pd.DataFrame()

for model in ["LMUPDRS", "LM"]:
    for idx, name_cross_val in enumerate(
        [
            f"{model}_performance_leave_1_cohort_out_RMAP.npy",
            f"{model}_performance_leave_1_sub_out_across_coh_RMAP.npy",
            f"{model}_performance_leave_1_sub_out_within_coh_RMAP.npy",
        ]
    ):

        p = np.load(
            os.path.join(PATH_CROSS_VAL_BASE, name_cross_val), allow_pickle="TRUE"
        ).item()

        for cohort_test in list(p.keys()):
            for sub_test in list(p[cohort_test].keys()):
                if p[cohort_test][sub_test] == {}:
                    continue
                df = df.append(
                    {
                        "performance_test": p[cohort_test][sub_test].score_test[0],
                        "cohort_test": cohort_test,
                        "sub_test": sub_test,
                        "model": model,
                        "cross_val_type": [
                            "leave one cohort out",
                            "leave one subject out across cohorts",
                            "leave one subject out within cohorts",
                        ][idx],
                    },
                    ignore_index=True,
                )

nm_plots.plot_df_subjects(
    df=df,
    x_col="cross_val_type",
    y_col="performance_test",
    title="Cross Validation Performances",
    hue="model",
    PATH_SAVE=os.path.join("figure", "cross_val_lm_updrs_model_comp.pdf"),
)
