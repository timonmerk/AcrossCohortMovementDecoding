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

    PATH_ML_MODEL = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"

    df_per_comp = pd.DataFrame()

    for cohort in [
        "Washington",
        "Berlin",
        "Pittsburgh",
        "Beijing",
    ]:

        PATH_FIND = os.path.join(PATH_ML_MODEL, cohort)
        files = glob.glob(f"{PATH_FIND}\\**\\*LM_ML_RES.p", recursive=True)
        for full_path in files:
            rec = os.path.normpath(full_path).split(os.sep)[8]
            name_ = os.path.basename(full_path)
            sub = name_[: name_.find("_ses-")] if cohort != "Washington" else name_[:2]

            # \sub-002_ses-EcogLfpMedOff02_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg\sub-002_ses-EcogLfpMedOff02_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg_LM_ML_RES.p"

            with open(full_path, "rb") as input:
                ML_res = cPickle.load(input)

            chs = list(ML_res.ch_ind_results.keys())

            for ch in chs:
                if len(ML_res.ch_ind_results[ch]["score_train"]) == 0:
                    continue
                y_true = np.concatenate(ML_res.ch_ind_results[ch]["y_test"])
                y_test_pr = np.concatenate(ML_res.ch_ind_results[ch]["y_test_pr"])

                ba_sample_wise = metrics.balanced_accuracy_score(
                    y_true, y_test_pr[:, 1] > 0.5
                )

                window_size = 10
                weights = np.repeat(1.0, window_size) / window_size

                # Calculate moving average using np.convolve
                y_pr_moving_avg = np.convolve(y_test_pr[:, 1], weights, "valid")[::10]

                y_true_moving_avg = np.convolve(y_true, weights, "valid")[
                    ::10
                ]  # y_true[9:][::10]

                ba_mean_1s = metrics.balanced_accuracy_score(
                    y_true_moving_avg > 0.5, y_pr_moving_avg > 0.5
                )

                # add here also the metric sensitivity, specificity and accuracy
                accuracy = metrics.accuracy_score(y_true, y_test_pr[:, 1] > 0.5)
                sensitivity = metrics.recall_score(y_true, y_test_pr[:, 1] > 0.5)
                specificity = metrics.recall_score(
                    y_true, y_test_pr[:, 1] > 0.5, pos_label=0
                )

                if df_per_comp.shape[1] == 0:
                    df_per_comp = pd.DataFrame(
                        {
                            "ba_mean_1s": ba_mean_1s,
                            "ba_sample_wise": ba_sample_wise,
                            "sub": sub,
                            "ch": ch,
                            "rec": rec,
                            "cohort": cohort,
                            "accuracy": accuracy,
                            "sensitivity": sensitivity,
                            "specificity": specificity,
                        },
                        index=[0],
                    )
                else:
                    df_per_comp = pd.concat(
                        [
                            df_per_comp,
                            pd.DataFrame(
                                {
                                    "ba_mean_1s": ba_mean_1s,
                                    "ba_sample_wise": ba_sample_wise,
                                    "sub": sub,
                                    "ch": ch,
                                    "rec": rec,
                                    "cohort": cohort,
                                    "accuracy": accuracy,
                                    "sensitivity": sensitivity,
                                    "specificity": specificity,
                                },
                                index=[0],
                            ),
                        ]
                    )

df_per_comp.reset_index().to_csv(
    "read_performances\\df_all_including_mean_1s_predictions_including_sens_spec_acc.csv"
)

df_rmap_best_ch = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")

# iterature through df_rmap_best_ch:
list_rows_best_rmap = []
for index, row in df_rmap_best_ch.iterrows():
    cohort = row["cohort"]
    sub = "sub-" + str(row["sub"]) if cohort != "Washington" else row["sub"]

    ch = row["ch"]
    df_q = df_per_comp.query("sub == @sub and cohort == @cohort and ch == @ch")
    list_rows_best_rmap.append(df_q)

df_comb = pd.concat(list_rows_best_rmap)


df_mean = df_comb.groupby(["sub", "cohort"]).mean().reset_index()

print(df_mean["ba_sample_wise"].mean())
print(df_mean["ba_mean_1s"].mean())

print(df_mean["accuracy"].mean())
print(df_mean["sensitivity"].mean())
print(df_mean["specificity"].mean())

df_mean.groupby(["cohort"]).mean()
df_mean.groupby(["cohort"]).std()

# combine the mean and std into a single df
df_mean = df_mean.groupby(["cohort"]).mean().reset_index()
df_std = df_comb.groupby(["cohort"]).std().reset_index()
# merge the two dataframes such that mean and std is rounded up to 2 decimal places
# and join the string with a plus minus sign
df_mean_std = df_mean.merge(df_std, on="cohort")
df_mean_std = df_mean_std.round(2)
df_mean_std["ba_sample_wise"] = (
    df_mean_std["ba_sample_wise_x"].astype(str)
    + " ± "
    + df_mean_std["ba_sample_wise_y"].astype(str)
)
df_mean_std["ba_mean_1s"] = (
    df_mean_std["ba_mean_1s_x"].astype(str)
    + " ± "
    + df_mean_std["ba_mean_1s_y"].astype(str)
)
# order the rows s.t. the cohorts are in order Berlin, Beijing, Pittsburgh, Washington
df_mean_std = df_mean_std.reindex([1, 0, 2, 3])
# save the dataframe only with the columns ba_sample_wise and ba_mean_1s
df_mean_std = df_mean_std[["cohort", "ba_sample_wise", "ba_mean_1s"]]
# save the dataframe to a csv file
df_mean_std.to_csv(
    "read_performances\\df_mean_performances_cohorts_1s_mean_vs_sample_wise_100ms.csv",
    index=False,
)


df_melted = df_mean.melt(
    id_vars=["sub", "cohort"],
    value_vars=["ba_sample_wise", "ba_mean_1s"],
    var_name="metric",
    value_name="performance_metric",
)

import seaborn as sns

df_melted["metric"] = df_melted["metric"].replace(
    {"ba_sample_wise": "sample wise", "ba_mean_1s": "1 sec average"}
)
df_melted["metric"] = df_melted["metric"].replace({"sample wise": "sample-wise"})


plt.figure(figsize=(5, 3), dpi=300)
sns.boxplot(
    data=df_melted,
    hue="metric",
    order=["Berlin", "Beijing", "Pittsburgh", "Washington"],
    y="performance_metric",
    x="cohort",
    palette="viridis",
)
plt.title("Sample-wise vs 1-sec averag performance comparison")
plt.ylabel("Balanced Accuracy")
plt.xlabel("Cohort")
plt.tight_layout()
plt.savefig("figure\\comoparison_single_channel_1s_vs_sample_wise.pdf")

plt.show()

# sns.swarmplot(data=df_melted, hue="metric", y="performance_metric", x="cohort", palette="viridis")
