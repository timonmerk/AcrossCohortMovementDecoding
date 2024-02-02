import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats, spatial

if __name__ == "__main__":
    # read df_all and average features per cohort
    ch_all = np.load("features_out_fft\\channel_all.npy", allow_pickle="TRUE").item()

    # select the channel
    df_best_rmap = pd.read_csv(
        r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv"
    )

    d_out = {}

    for cohort in ch_all.keys():
        for sub in ch_all[cohort].keys():
            # for ch in ch_all[cohort][sub].keys():
            ch_best = df_best_rmap.query("sub == @sub and cohort == @cohort").iloc[0][
                "ch"
            ]
            runs = list(ch_all[cohort][sub][ch_best].keys())

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

            # store the mean features for movement and rest
            d_out[f"{cohort}_{sub}"] = {}
            d_out[f"{cohort}_{sub}"]["mean_features_mov"] = dat_concat[
                np.where(lab_concat)[0], :
            ]  # .mean(axis=0)
            d_out[f"{cohort}_{sub}"]["mean_features_rest"] = dat_concat[
                np.where(np.logical_not(lab_concat))[0], :
            ]  # .mean(axis=0)

    # compute now for every feature the cosine similarity

    from sklearn import metrics

    spatial.distance.cosine(
        d_out["Berlin_001"]["mean_features_mov"].mean(axis=0),
        d_out["Berlin_002"]["mean_features_mov"].mean(axis=0),
    )

    matrix_mov = np.zeros([len(d_out.keys()), len(d_out.keys())])
    matrix_rest = np.zeros([len(d_out.keys()), len(d_out.keys())])
    matrix_diff = np.zeros([len(d_out.keys()), len(d_out.keys())])

    for sub in d_out.keys():
        for sub_ in d_out.keys():
            cos_sim = metrics.pairwise.cosine_similarity(
                d_out[sub]["mean_features_mov"].mean(axis=0).reshape(1, -1),
                d_out[sub_]["mean_features_mov"].mean(axis=0).reshape(1, -1),
            )
            matrix_mov[
                list(d_out.keys()).index(sub), list(d_out.keys()).index(sub_)
            ] = cos_sim
            cos_sim = metrics.pairwise.cosine_similarity(
                d_out[sub]["mean_features_rest"].mean(axis=0).reshape(1, -1),
                d_out[sub_]["mean_features_rest"].mean(axis=0).reshape(1, -1),
            )
            matrix_rest[
                list(d_out.keys()).index(sub), list(d_out.keys()).index(sub_)
            ] = cos_sim

            # subtract the move and rest and take the absolute value and then run the cosine similarity
            cos_sim = metrics.pairwise.cosine_similarity(
                np.abs(
                    d_out[sub]["mean_features_mov"].mean(axis=0)
                    - d_out[sub]["mean_features_rest"].mean(axis=0)
                ).reshape(1, -1),
                np.abs(
                    d_out[sub_]["mean_features_mov"].mean(axis=0)
                    - d_out[sub_]["mean_features_rest"].mean(axis=0)
                ).reshape(1, -1),
            )
            matrix_diff[
                list(d_out.keys()).index(sub), list(d_out.keys()).index(sub_)
            ] = cos_sim

    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(matrix_mov, aspect="auto")
    plt.title("Cosine similarity Move")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(matrix_rest, aspect="auto")
    plt.title("Cosine similarity Rest")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(matrix_diff, aspect="auto")
    plt.title("Cosine similarity Vector Diff")
    plt.colorbar()
    plt.tight_layout()

    # plot the move and rest features for every channel

    plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.imshow(
        np.array([d_out[f]["mean_features_mov"].mean(axis=0) for f in d_out.keys()]).T,
        aspect="auto",
        cmap="coolwarm",
    )
    plt.title("Move")
    plt.gca().invert_yaxis()
    plt.yticks(
        np.arange(7),
        ["theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA"],
    )
    plt.xlabel("Subjects")
    plt.xticks(
        np.arange(len(d_out.keys())), list(d_out.keys()), rotation=90, fontsize=6
    )
    plt.colorbar()
    plt.clim(-0.5, 0.5)
    plt.subplot(1, 2, 2)
    plt.imshow(
        np.array([d_out[f]["mean_features_rest"].mean(axis=0) for f in d_out.keys()]).T,
        aspect="auto",
        cmap="coolwarm",
    )
    plt.title("Rest")
    plt.gca().invert_yaxis()
    plt.yticks(
        np.arange(7),
        ["theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA"],
    )
    plt.xlabel("Subjects")
    plt.colorbar()
    plt.clim(-0.5, 0.5)
    plt.xticks(
        np.arange(len(d_out.keys())), list(d_out.keys()), rotation=90, fontsize=6
    )
    plt.tight_layout()

    cos_sim = np.zeros([len(d_out.keys()), len(d_out.keys())])
