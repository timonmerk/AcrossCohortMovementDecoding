import os
import mne_bids
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from bids import BIDSLayout
from py_neuromodulation import (
    nm_across_patient_decoding,
    nm_analysis,
    nm_decode,
    nm_plots,
)
from sklearn import linear_model, metrics
from multiprocessing import Pool
import _pickle as cPickle
import glob


if __name__ == "__main__":

    # new idea: train using ch_all.npy a linear model for every channel
    # and store the model coefficients of a linear model for every channel

    PATH_ML_MODEL = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"
    ch_all = np.load(
        os.path.join(PATH_ML_MODEL, "channel_all.npy"), allow_pickle=True
    ).item()

    df_best_rmap = pd.read_csv(
        r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv"
    )

    # initialize the dataframe with columns of the frequency bands
    df_out = pd.DataFrame(columns=["cohort", "sub", "theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA"])

    for cohort in ch_all.keys():
        for sub in ch_all[cohort].keys():
            # for ch in ch_all[cohort][sub].keys():
            ch_best = df_best_rmap.query("sub == @sub and cohort == @cohort").iloc[0][
                "ch"
            ]
            runs = list(ch_all[cohort][sub][ch_best].keys())
            runs = [r for r in runs if "StimOn" not in r]
            if len(runs) > 1:
                dat_concat = np.concatenate(
                    [ch_all[cohort][sub][ch_best][run]["data"] for run in runs], axis=0
                )
                lab_concat = np.concatenate(
                    [ch_all[cohort][sub][ch_best][run]["label"] for run in runs], axis=0
                )
            else:
                dat_concat = ch_all[cohort][sub][ch_best][runs[0]]["data"]
                lab_concat = ch_all[cohort][sub][ch_best][runs[0]]["label"]

            model = linear_model.LogisticRegression(class_weight="balanced").fit(
                dat_concat, lab_concat
            )

            # save the model coefficients
            df_out = df_out.append(
                {
                    "cohort": cohort,
                    "sub": sub,
                    "theta": model.coef_[0][0],
                    "alpha": model.coef_[0][1],
                    "low beta": model.coef_[0][2],
                    "high beta": model.coef_[0][3],
                    "low gamma": model.coef_[0][4],
                    "high gamma": model.coef_[0][5],
                    "HFA": model.coef_[0][6],
                },
                ignore_index=True,
            )


    # use seaborn to plot the absolute sum of the coefficients
    import seaborn as sb
    import matplotlib.pyplot as plt
    
    #df_plt = df_out.groupby("cohort").sum()
    df_plt = df_out.melt(id_vars="cohort", var_name="frequency_band", value_name="coef_sum").query("frequency_band != 'sub'")
    plt.figure(figsize=(7, 4), dpi=300)
    hue_order = ["Berlin", "Beijing", "Pittsburgh", "Washington"]
    sb.boxplot(x="frequency_band", y="coef_sum", data=df_plt, palette="viridis", hue="cohort", hue_order=hue_order)
    plt.xlabel("Frequency bands")
    plt.ylabel("Mean band power z-scored [a.u.]")
    plt.xticks(rotation=90)
    # place the legend to the right outside the plot
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Movement decoding model coefficients")
    plt.tight_layout()

    df_plt = df_out.groupby("cohort").sum().reset_index()
    df_plt = df_plt.melt(id_vars="cohort", var_name="frequency_band", value_name="coef_sum").query("frequency_band != 'sub'")
    plt.figure(figsize=(7, 4), dpi=300)
    sb.barplot(x="frequency_band", y="coef_sum", data=df_plt, palette="viridis", hue="cohort", hue_order=hue_order)
    plt.xlabel("Frequency bands")
    plt.ylabel("Mean band power z-scored [a.u.]")
    plt.xticks(rotation=90)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title("Movement decoding model coefficients")
    plt.tight_layout()



    # transform the dataframe s.t. each column is stored in a frequency_band column
    df_plt = df_out.melt(id_vars="sub", var_name="frequency_band", value_name="coef_sum").query("frequency_band != 'cohort'")
    plt.figure(figsize=(7, 4), dpi=300)
    sb.boxplot(y="coef_sum", x="frequency_band", data=df_plt, palette="viridis")
    sb.swarmplot(y="coef_sum", x="frequency_band", data=df_plt, palette="viridis")
    plt.xticks(rotation=90)
    plt.xlabel("Frequency bands")
    plt.ylabel("Mean band power z-scored [a.u.]")
    plt.title("Movement decoding model coefficients")
    plt.tight_layout()




    plt.figure(figsize=(7, 4), dpi=300)
    df_plt = df_out.select_dtypes(include=[np.number]).abs().sum()
    sb.barplot(x=df_plt.index, y=df_plt.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.xlabel("Frequency bands")
    plt.ylabel("Mean band power z-scored [a.u.]")
    plt.title("Sum of absolute values model coefficients")
    plt.tight_layout()
    # transform the dataframe s.t. each column is stored in a frequency_band column

    plt.figure(figsize=(7, 4), dpi=300)
    sb.barplot(y="coef_sum", x="frequency_band", data=df_plt, palette="viridis")
    sb.swarmplot(y="coef_sum", x="frequency_band", data=df_plt, palette="viridis")
    plt.title("Movement decoding model coefficients")
    plt.tight_layout()