import numpy as np
import os
import pandas as pd
from sklearn import linear_model, metrics, model_selection

from py_neuromodulation import nm_analysis
from matplotlib import pyplot as plt


ch_all = np.load(
    os.path.join("features_out_fft", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

df_regions = pd.read_csv(
    os.path.join("read_performances", "df_ch_performances_including_brain_regions.csv")
)

subs = ch_all["Berlin"].keys()
df_res = pd.DataFrame(columns=["sub", "ch", "ba", "region"])

precentral_y_predict = []
precentral_y_true = []
postcentral_y_predict = []
postcentral_y_true = []

for sub in subs:
    for ch in ch_all["Berlin"][sub].keys():
        data_concat = []
        label_concat = []
        for recs in ch_all["Berlin"][sub][ch].keys():
            data_concat.append(ch_all["Berlin"][sub][ch][recs]["data"])
            label_concat.append(ch_all["Berlin"][sub][ch][recs]["label"])
        data_concat = np.concatenate(data_concat)
        label_concat = np.concatenate(label_concat)

        # first remove the movement labels
        # for every instance where label is 1 set the previous 5 values 2
        # then remove all instances where label is 2
        label_concat = label_concat.astype(int)
        diffs_ = np.diff(label_concat)
        idx_ = np.where(diffs_ == 1)[0]
        for idx in idx_:
            if idx >= 0 and idx < label_concat.shape[0]:
                label_concat[idx - 20 : idx + 1] = 2

        idx_select = np.where(label_concat != 1)[0]
        data_concat = data_concat[idx_select, :]
        label_concat = label_concat[idx_select] > 0

        # run a 3 fold non-shuffled Kfold cross validation and report the mean balanced accuracy
        model = linear_model.LogisticRegression(class_weight="balanced")
        cv = model_selection.StratifiedKFold(n_splits=3)
        predict = model_selection.cross_val_predict(
            model,
            data_concat,
            label_concat,
            cv=cv,
        )

        scores = metrics.balanced_accuracy_score(label_concat, predict)

        # select the brain region where the cohort is Berlin with the right subject and channel
        df_regions.query("sub == @sub and ch == @ch and cohort == 'Berlin'")
        df_regions_sub = df_regions.loc[
            (df_regions["sub"] == sub)
            & (df_regions["ch"] == ch)
            & (df_regions["cohort"] == "Berlin")
        ]
        # check if it's avaliable, if not continue
        if df_regions_sub.shape[0] == 0:
            continue

        region = df_regions_sub["region"].values[0]

        # use the concat function of pandas to add the results to the dataframe
        df_res = pd.concat(
            [
                df_res,
                pd.DataFrame(
                    {
                        "sub": [sub],
                        "ch": [ch],
                        "ba": [np.mean(scores)],
                        "region": [region],
                    }
                ),
            ]
        )

        if "Precentral" in region:
            epochs_pr_mc, epochs_true_mc = nm_analysis.Feature_Reader.get_epochs(
                np.expand_dims(predict, axis=(1, 2)), label_concat, 7, 10, 0.1
            )

            precentral_y_predict.append(
                np.squeeze(np.mean(epochs_pr_mc, axis=0).T)[:55]
            )
            # precentral_y_true.append(label_concat)
        elif "Postcentral" in region:
            epochs_pr_sc, epochs_true_sc = nm_analysis.Feature_Reader.get_epochs(
                np.expand_dims(predict, axis=(1, 2)), label_concat, 7, 10, 0.1
            )
            postcentral_y_predict.append(
                np.squeeze(np.mean(epochs_pr_sc, axis=0).T)[:55]
            )
            # postcentral_y_true.append(label_concat)


plt.figure(figsize=(3.5, 3), dpi=300)
time_ = np.arange(-5.5, 0, 0.1)

for i in range(len(precentral_y_predict)):
    plt.plot(time_, precentral_y_predict[i], color="#45a778", alpha=0.2)
for i in range(len(postcentral_y_predict)):
    plt.plot(time_, postcentral_y_predict[i], color="#3c6682", alpha=0.2)

plt.plot(
    time_,
    np.squeeze(np.mean(epochs_pr_mc, axis=0).T)[:55],
    label="motor cortex",
    color="#45a778",
    linewidth=3,
)
plt.plot(
    time_,
    np.squeeze(np.mean(epochs_pr_sc, axis=0).T)[:55],
    label="somatosensory cortex",
    color="#3c6682",
    linewidth=3,
)
plt.legend()
plt.xlabel("Time till movement onset[s]")
plt.ylabel("Movement intention prediction")
plt.title("Movement intention prediction")
plt.tight_layout()
plt.savefig("figure\\mean_prediction_movement_intention.pdf")
plt.show()


epochs_pr_mc, epochs_true_mc = nm_analysis.Feature_Reader.get_epochs(
    np.expand_dims(np.concatenate(precentral_y_predict), axis=(1, 2)),
    np.concatenate(precentral_y_true),
    7,
    10,
    0.2,
)

epochs_pr_sc, epochs_true_sc = nm_analysis.Feature_Reader.get_epochs(
    np.expand_dims(np.concatenate(postcentral_y_predict), axis=(1, 2)),
    np.concatenate(postcentral_y_true),
    7,
    10,
    0.2,
)


plt.figure()
plt.plot(np.squeeze(np.mean(epochs_pr_mc, axis=0).T)[:35], label="motor cortex")
plt.plot(np.squeeze(np.mean(epochs_pr_sc, axis=0).T)[:35], label="somatosensory cortex")
plt.legend()
plt.show()

# select only Precentral_R,  Postcentral_R, Precentral_L,  Postcentral_L
df_res = df_res.query(
    "region == 'Precentral_R' or region == 'Postcentral_R' or region == 'Precentral_L' or region == 'Postcentral_L'"
)
import seaborn as sns

# average Precentral_R and Precentral_L; and Postcentral_R and Postcentral_L
df_res = df_res.replace("Precentral_L", "Precentral")
df_res = df_res.replace("Precentral_R", "Precentral")
df_res = df_res.replace("Postcentral_L", "Postcentral")
df_res = df_res.replace("Postcentral_R", "Postcentral")

# plot with a swarmplot and a boxplot on top, x axis region
plt.figure(figsize=(3.5, 3), dpi=300)

ax = sns.swarmplot(
    x="region",
    y="ba",
    data=df_res,
    color=".25",
)
ax = sns.boxplot(x="region", y="ba", data=df_res, palette="viridis")
ax.set_xlabel("Region")
ax.set_ylabel("Balanced Accuracy")
ax.set_title("Movement intention performance")
plt.tight_layout()
plt.savefig("figure\\Mean_performance_Berlin_movement_intention.pdf")
plt.show()
