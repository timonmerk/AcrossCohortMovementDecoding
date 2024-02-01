import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from py_neuromodulation import nm_analysis
from scipy import stats

if __name__ == "__main__":
    # read df_all and average features per cohort
    channel_all = np.load(
        "features_out_fft\\channel_all.npy", allow_pickle="TRUE"
    ).item()

    # select the channel
    df_best_rmap = pd.read_csv(
        r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv"
    )

    def get_data_sub_ch(channel_all, cohort, sub, ch):
        X_train = []
        y_train = []

        for f in channel_all[cohort][sub][ch].keys():
            X_train.append(channel_all[cohort][sub][ch][f]["data"])
            y_train.append(channel_all[cohort][sub][ch][f]["label"])
        if len(X_train) > 1:
            X_train = np.concatenate(X_train, axis=0)
            y_train = np.concatenate(y_train, axis=0)
        else:
            X_train = X_train[0]
            y_train = y_train[0]

        return X_train, y_train

    def get_data_channels(sub_test: str, cohort_test: str, df_rmap: list):
        ch_test = df_rmap.query("cohort == @cohort_test and sub == @sub_test")[
            "ch"
        ].iloc[0]
        X_test, y_test = get_data_sub_ch(channel_all, cohort_test, sub_test, ch_test)
        return X_test, y_test

    avg_epochs = []
    for cohort in channel_all.keys():
        X_train_comb = []
        y_train_comb = []
        for sub in channel_all[cohort].keys():
            X, y = get_data_channels(sub, cohort, df_rmap=df_best_rmap)

            # potentially get here the movement averaged features
            X_, y_ = nm_analysis.Feature_Reader.get_epochs(
                np.expand_dims(X, axis=1), y, epoch_len=3, sfreq=10, threshold=0.5
            )
            X_train_comb.append(np.squeeze(X_).mean(axis=0))
        avg_epochs.append(np.array(X_train_comb).mean(axis=0))

        # plot here for all cohorts the averages

    plt.figure(figsize=(10, 7), dpi=300)
    for idx, cohort in enumerate(channel_all.keys()):
        plt.subplot(2, 2, idx + 1)
        plt.imshow(stats.zscore(avg_epochs[idx].T, axis=1), aspect="auto")
        plt.gca().invert_yaxis()
        plt.title(cohort)
        # plt.clim(-0.5, 0.5)
        plt.colorbar()
        num_ = avg_epochs[0].shape[0]
        plt.xticks(
            np.arange(num_)[::3],
            np.round(np.arange(-(num_ / 10) / 2, (num_ / 10) / 2, 0.1), 1)[::3],
        )
        plt.yticks(
            np.arange(avg_epochs[0].shape[1]),
            [
                r"$\theta$",
                r"$\alpha$",
                r"$l\beta$",
                r"$h\beta$",
                r"$l\gamma$",
                r"$h\gamma$",
                r"$HFA$",
            ],
        )
        plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig("figure\\cohort_averaged_fft_features.pdf")
