import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats, nm_across_patient_decoding, nm_decode, nm_plots

COMPARE_MEAN_PER = True
if COMPARE_MEAN_PER is True:
    dfs_ = []
    for RUN_PARRM_STIM_DATA in [True, False]:
        if RUN_PARRM_STIM_DATA is True:
            df_1 = pd.read_csv(
                "stim_off_on_prediction\\df_STIM_ON_OFF_predict_parrm.csv"
            )
            df_2 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_parrm.csv")
        else:
            df_1 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_predict.csv")
            df_2 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF.csv")
        df_comb = pd.concat([df_1, df_2])
        df_comb["artifact_rejection_method"] = (
            "PARRM" if RUN_PARRM_STIM_DATA else "None"
        )
        dfs_.append(df_comb)

    # add here the bandstop filtered option
    df_1 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_predict_bandstop.csv")
    df_2 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_bandstop.csv")
    df_comb = pd.concat([df_1, df_2])
    df_comb["artifact_rejection_method"] = "Bandstop Filtering"
    dfs_.append(df_comb)

    print(pd.concat(dfs_).groupby(["Model Type", "artifact_rejection_method"]).mean())

# plot here the grouped boxplot:
df_plt = pd.concat(dfs_)

plt.figure(figsize=(10, 6), dpi=300)
order_ = [
    "STIM OFF",
    "STIM ON",
    "STIM OFF->ON Predict",
    "STIM ON->OFF Predict",
    "STIM ON-OFF->OFF Predict",
    "STIM ON-OFF->ON Predict",
]
hue_order = ["None", "PARRM", "Bandstop Filtering"]

ax = sb.boxplot(
    x="Model Type",
    y="Test Performance",
    order=order_,
    hue="artifact_rejection_method",
    data=df_plt,
    dodge=True,
    palette="viridis",
    hue_order=hue_order,
)
n_hues = 3
handles, labels = ax.get_legend_handles_labels()
l = plt.legend(
    handles[0:n_hues],
    labels[0:n_hues],
    bbox_to_anchor=(1.05, 1),
    loc=2,
    title="Artifact rejection method",
    borderaxespad=0.0,
)
sb.swarmplot(
    x="Model Type",
    y="Test Performance",
    order=order_,
    hue="artifact_rejection_method",
    data=df_plt,
    dodge=True,
    palette="viridis",
    s=5,
    hue_order=hue_order,
)

plt.xticks(rotation=90)
plt.title("Stim ON / OFF Performances with different artifact rejection methods")
plt.tight_layout()
plt.savefig(os.path.join("figure", "stim_on_off_art_reject_none_parrm_bandpass.pdf"))


RUN_PARRM_STIM_DATA = True
if RUN_PARRM_STIM_DATA is True:
    df_1 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_predict_parrm.csv")
    df_2 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_parrm.csv")
else:
    df_1 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_predict.csv")
    df_2 = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF.csv")
df_comb = pd.concat([df_1, df_2])

df = df_comb
x_col = "Model Type"
y_col = "Test Performance"
if RUN_PARRM_STIM_DATA is True:
    PATH_SAVE = os.path.join("figure", "stim_off_on_comp_predict_all_boxplot_parrm.pdf")
else:
    PATH_SAVE = os.path.join("figure", "stim_off_on_comp_predict_all_boxplot.pdf")

hue = None
order_ = [
    "STIM OFF",
    "STIM ON",
    "STIM OFF->ON Predict",
    "STIM ON->OFF Predict",
    "STIM ON-OFF->OFF Predict",
    "STIM ON-OFF->ON Predict",
]

alpha_box = 0.4
plt.figure(figsize=(5, 3), dpi=300)
sb.boxplot(
    x=x_col,
    y=y_col,
    hue=hue,
    data=df,
    palette="viridis",
    showmeans=False,
    boxprops=dict(alpha=alpha_box),
    showcaps=True,
    showbox=True,
    showfliers=False,
    order=order_,
    notch=False,
    whiskerprops={"linewidth": 2, "zorder": 10, "alpha": alpha_box},
    capprops={"alpha": alpha_box},
    medianprops=dict(linestyle="-", linewidth=5, color="gray", alpha=alpha_box),
)

ax = sb.stripplot(
    x=x_col,
    y=y_col,
    hue=hue,
    data=df,
    order=order_,
    palette="viridis",
    dodge=True,
    s=5,
)

if hue is not None:
    n_hues = df[hue].nunique()

    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(
        handles[0:n_hues],
        labels[0:n_hues],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        title=hue,
        borderaxespad=0.0,
    )
plt.title("Stim ON / OFF Specific Performances")
plt.ylabel(y_col)
plt.xticks(rotation=90)
if PATH_SAVE is not None:
    plt.savefig(
        PATH_SAVE,
        bbox_inches="tight",
    )

plt.figure(figsize=(6, 5), dpi=300)
sb.barplot(
    data=df_comb,
    x="Subject",
    y="Test Performance",
    hue="Model Type",
    palette="viridis",
)
plt.ylim(
    0.5,
)
plt.legend(bbox_to_anchor=(1.04, 1))
plt.title(f"STIM ON / OFF Comparison Berlin subjects")
plt.savefig(
    os.path.join("figure", "stim_off_on_comp_predict_all.pdf"),
    bbox_inches="tight",
)
