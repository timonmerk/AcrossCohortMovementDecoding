import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn import metrics, linear_model, model_selection

import py_neuromodulation
from py_neuromodulation import nm_decode, nm_RMAP, nm_plots, nm_stats

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
                        "Movement Detection Rate" : p[cohort_test][sub_test].mov_detection_rates_test[0],
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

# print performances
print(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"].std())

print(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"].std())

print(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"].std())

print(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"].std())

print(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "GridPoints")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "GridPoints")]["Test Performance"].std())

print(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "RMAP")]["Test Performance"].mean())
print(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "RMAP")]["Test Performance"].std())

# permutation tests against balanced accuracy 0.5 chance level

# leave one subject out within cohorts
x=np.array(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))

x=np.array(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))


# leave one subject out across cohorts
x=np.array(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))

x=np.array(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))

# leave one cohort out
x=np.array(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "RMAP")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))

x=np.array(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "GridPoints")]["Test Performance"])
y=np.repeat(0.5, x.shape[0])
print(nm_stats.permutationTest_relative(y, x, False, p=5000))

nm_stats.permutationTest_relative(
    np.array(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"]),
    np.array(df[(df["Cross Validation Type"] == "leave one subject out across cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"]),
    False,
    p=5000
)
nm_stats.permutationTest_relative(
    np.array(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "RMAP")]["Test Performance"]),
    np.array(df[(df["Cross Validation Type"] == "leave one subject out within cohorts") & (df["Model Type"] == "GridPoints")]["Test Performance"]),
    False,
    p=5000
)
nm_stats.permutationTest_relative(
    np.array(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "RMAP")]["Test Performance"]),
    np.array(df[(df["Cross Validation Type"] == "leave one cohort out") & (df["Model Type"] == "GridPoints")]["Test Performance"]),
    False,
    p=5000
)

nm_plots.plot_df_subjects(
    df=df.iloc[::-1],
    x_col="Cross Validation Type",
    y_col="Test Performance",
    title="Cross Validation Performances",
    hue="Model Type",
    PATH_SAVE=os.path.join("figure", f"cross_val_model_comparison_1timestep.pdf"),
)

nm_plots.plot_df_subjects(
    df=df.iloc[::-1],  # flip s.t. cross validation approach get's displayed correctly
    x_col="Cross Validation Type",
    y_col="Movement Detection Rate",
    title="Cross Validation Performances",
    hue="Model Type",
    PATH_SAVE=os.path.join("figure", f"cross_val_model_comparison_1timestep_Move_detection_Rate.pdf"),
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
    df_rmap_corr_with_det = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")

    per_per = np.zeros([38, 38])
    per_per_det = np.zeros([38, 38])
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

                    per_insert_det = p[cohort_test][sub_test][cohort_train][
                        sub_train
                    ].mov_detection_rates_test[0]

                    if cohort_test == cohort_train and sub_test == sub_train:
                        per_insert = df_rmap_corr.query(
                            "cohort == @cohort_train and sub == @sub_train"
                        )["performance_test"].iloc[0]
                        ch_select = df_rmap_corr.query(
                            "cohort == @cohort_train and sub == @sub_train"
                        )["ch"].iloc[0]
                        per_insert_det = df_rmap_corr_with_det.query(
                            "cohort == @cohort_train and sub == @sub_train and ch == @ch_select"
                        )["mov_detection_rates_test"].iloc[0]

                    per_per[idx_train, idx_test] = per_insert
                    per_per_det[idx_train, idx_test] = per_insert_det

                    df = df.append(
                        {
                            "cohort_test": cohort_test,
                            "sub_test": sub_test,
                            "cohort_train": cohort_train,
                            "sub_train": sub_train,
                            "Performance Test": per_insert,
                            "Mov_detection" : per_insert_det,
                            "Model Type": model_sel_name,
                        },
                        ignore_index=True,
                    )

                    idx_train += 1
            idx_test += 1
            idx_train = 0

    np.save(f"p2p_arr_{model_name}.npy", per_per)
    per_per = np.load(f"p2p_arr_{model_name}.npy")

    sub_list = []
    for cohort in list(p.keys()):
        for sub in list(p[cohort].keys()):
            sub_list.append(f"{cohort}_{sub}")

    # per_per = np.load(os.path.join("figure", "p2p_arr.npy"), allow_pickle="TRUE")

    # get the average training performances balanced accuracy: (and point out sub002 in paper)
    avg_ = []
    for i in range(38):
        print(f"{sub_list[i]} {np.round(np.mean(per_per[:, i]), 2)} {np.round(np.std(per_per[:, i]), 2)}")
        avg_.append(np.mean(per_per[:, i]))
    print(f"overall average: {np.mean(avg_)} pm {np.std(avg_)}")

    # get the average training performances balanced accuracy: (and point out sub002 in paper)
    avg_col = []
    avg_row = []
    for i in range(38):
        print(f"{sub_list[i]} {np.round(np.mean(per_per_det[:, i]), 2)} {np.round(np.std(per_per_det[:, i]), 2)}")
        avg_col.append(np.mean(per_per_det[:, i]))
        avg_row.append(np.mean(per_per_det[i, :]))
    print(f"overall average: {np.mean(avg_)} pm {np.std(avg_)}")

    def get_sorted_arr(arr):
        sort_ax0 = np.mean(arr, axis=0)
        sort_ax1 = np.mean(arr, axis=1)

        arr_s_ax0 = arr[:, np.argsort(sort_ax0)]

        arr_s_ax1 = arr_s_ax0[np.argsort(sort_ax1), :]

        return arr_s_ax1, np.argsort(sort_ax0), np.argsort(sort_ax1)

    pps, s0, s1 = get_sorted_arr(per_per)

    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(pps[::-1, :], aspect="auto")
    plt.yticks(np.arange(len(sub_list)), np.array(sub_list)[s1])
    plt.xticks(np.arange(len(sub_list)), np.array(sub_list)[s0], rotation=90)
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

    # order plot based on average prediction performance
    idx_order_avg_row = np.argsort(avg_row)[::-1]
    idx_order_avg_col = np.argsort(avg_col)[::-1]
    
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(per_per[idx_order_avg_col, :][:,idx_order_avg_row], aspect="auto")  # 
    plt.xticks(np.arange(len(sub_list)), np.array(sub_list)[idx_order_avg_row], rotation=90)
    plt.yticks(np.arange(len(sub_list)), np.array(sub_list)[idx_order_avg_col])
    cbar = plt.colorbar()
    cbar.set_label("Balanced Accuracy")
    plt.title(f"{model_name} patient to patient predictions")
    plt.xlabel("Train Patient")
    plt.ylabel("Test Patient")
    plt.clim(
        0.5,
    )
    plt.tight_layout()
    plt.savefig(
        f"p2p_{model_name}_ordered_by_per.pdf",
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



arr_orig = np.arange(25).reshape((5, 5))

arr_orig = np.array([[4, 2, 1],
                [5, 4, 2],
                [7, 6, 5]])

arr_s1 = np.sort(per_per, axis=1)

# sort by columns
arr_s2 = np.transpose(np.sort(np.transpose(arr_s1)))

plt.imshow(arr_s2)


# first step: sort columns
# 2nd step: sort rows

per_avg_ax1 = per_per.mean(axis=0)
per_avg_ax0 = per_per.mean(axis=1)

# debug: first ax0
per_sort_ax0 = per_per[np.argsort(per_avg_ax0), :]

per_sort_ax1 = per_sort_ax0[:, np.argsort(per_avg_ax1)]

plt.imshow(per_sort_ax1)

arr = np.array([[1, 5, 3, 6], [4, 7, 6, 1], [0, 9, 2, 8]])


pps, s0, s1 = get_sorted_arr(per_per)

pps_, s0_, s1_ = get_sorted_arr(arr)

