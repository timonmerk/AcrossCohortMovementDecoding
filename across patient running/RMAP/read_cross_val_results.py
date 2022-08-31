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

# [
#        "XGB_performance_leave_1_cohort_out_RMAP.npy",
#        "XGB_performance_leave_1_sub_out_within_coh_RMAP.npy",
# ]

for model in ["LMGridPoints", "LM"]:
    if model == "LM":
        model_sel_name = "RMAP"
    else:
        model_sel_name = "GridPoints"
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
                df = df.append(
                    {
                        "Test Performance": p[cohort_test][sub_test].score_test[0],
                        "cohort_test": cohort_test,
                        "sub_test": sub_test,
                        "Model Type": model_sel_name,
                        "Cross Validation Type": [
                            "leave one cohort out",
                            "leave one subject out across cohorts",
                            "leave one subject out within cohorts",
                        ][idx],
                    },
                    ignore_index=True,
                )

nm_plots.plot_df_subjects(
    df=df,
    x_col="Cross Validation Type",
    y_col="Test Performance",
    title="Cross Validation Performances",
    hue="Model Type",
    PATH_SAVE=os.path.join("figure", f"cross_val_model_comparison_1timestep.pdf"),
)

nm_plots.plot_df_subjects(
    df=df,
    x_col="cross_val_type",
    y_col="performance_test",
    title="Cross Validation Performances",
    hue="cohort_test",
    PATH_SAVE=os.path.join("figure", f"cross_val_{model}_1timestep.pdf"),
)

# 2. p2p predictions

df = pd.DataFrame()
for model_name in ["LMGridPoints", "LM"]:  # ["LM", "XGB"]:
    if model_name == "LM":
        model_sel_name = "RMAP"
    else:
        model_sel_name = "GridPoints"
    p = np.load(
        os.path.join(
            PATH_CROSS_VAL_BASE, f"{model_name}_performance_p2p_RMAP.npy"
        ),  # LM
        allow_pickle="TRUE",
    ).item()

    # load as well the df to fill the patient individual performances
    df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

    per_per = np.zeros([38, 38])
    idx_train = 0
    idx_test = 0
    # df = pd.DataFrame() # seelct this for individual plot
    for cohort_test in list(p.keys()):
        for sub_test in list(p[cohort_test].keys()):
            for cohort_train in list(p[cohort_test][sub_test].keys()):
                for sub_train in list(p[cohort_test][sub_test][cohort_train].keys()):

                    per_insert = p[cohort_test][sub_test][cohort_train][
                        sub_train
                    ].score_test[0]
                    if cohort_test == cohort_train and sub_test == sub_train:
                        per_insert = df_rmap_corr.query(
                            "cohort == @cohort_train and sub == @sub_train"
                        )["performance_test"].iloc[0]
                    per_per[idx_train, idx_test] = per_insert

                    df = df.append(
                        {
                            "cohort_test": cohort_test,
                            "sub_test": sub_test,
                            "cohort_train": cohort_train,
                            "sub_train": sub_train,
                            "Performance Test": per_insert,
                            "Model Type": model_sel_name,
                        },
                        ignore_index=True,
                    )

                    idx_train += 1
            idx_test += 1
            idx_train = 0

    np.save(f"p2p_arr_{model_name}.npy", per_per)

    sub_list = []
    for cohort in list(p.keys()):
        for sub in list(p[cohort].keys()):
            sub_list.append(f"{cohort}_{sub}")

    # per_per = np.load(os.path.join("figure", "p2p_arr.npy"), allow_pickle="TRUE")

    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(per_per, aspect="auto")
    plt.xticks(np.arange(len(sub_list)), sub_list, rotation=90)
    plt.yticks(np.arange(len(sub_list)), sub_list)
    cbar = plt.colorbar()
    cbar.set_label("Balanced Accuracy")
    plt.title(f"{model_name} patient to patient predictions")
    plt.xlabel("Train Patient")
    plt.ylabel("Test Patient")
    plt.clim(
        0.5,
    )
    plt.savefig(
        f"p2p_{model_name}.pdf",
        bbox_inches="tight",
    )


nm_plots.plot_df_subjects(
    df=df,
    x_col="Model Type",
    y_col="Performance Test",
    title="P2P Cross Validation Performances",
    hue=None,
    PATH_SAVE=os.path.join("figure", f"p2p_comp_model_type_comparison.pdf"),
)
