from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import shuffle
import pickle
import numpy as np
import os
import pandas as pd
import xgboost
from matplotlib import pyplot as plt

# select data from the best R-Map selected channel
OUTPATH = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"

ch_all = np.load(
    os.path.join(OUTPATH, "channel_all.npy"),
    allow_pickle="TRUE",
).item()

df = pd.read_csv("read_performances\\df_ch_performances.csv")

dat_out = []
label_out = []
dat_out_ON = []
label_out_ON = []

subs_ = []

idx_ = 0
for cohort in df["cohort"].unique():
    if cohort != "Berlin":
        continue
    for sub in df.query("cohort == @cohort")["sub"].unique():
        df_sub = df.query("sub == @sub and cohort == @cohort")
        ch = str(df_sub.query('performance_test == performance_test.max()')["ch"].iloc[0])

        recs = [r for r in list(ch_all[cohort][sub][ch].keys()) if "StimOn" not in r]
        for rec in recs:
            data = ch_all[cohort][sub][ch][rec]["data"]
            label = ch_all[cohort][sub][ch][rec]["label"]

            dat_out.append(data)
            label_out.append(label)

        recs = [r for r in list(ch_all[cohort][sub][ch].keys()) if "StimOn" in r]
        if len(recs) > 0:
            sub_dat = []
            sub_label = []
            for rec in recs:
                data_ON = ch_all[cohort][sub][ch][rec]["data"]
                label_ON = ch_all[cohort][sub][ch][rec]["label"]
                sub_dat.append(data_ON)
                sub_label.append(label_ON)
            dat_out_ON.append(np.concatenate(sub_dat))
            label_out_ON.append(np.concatenate(sub_label))

model_train = linear_model.LogisticRegression()

X_train = np.concatenate(dat_out)
y_train = np.concatenate(label_out)

model_train.fit(X_train, y_train)
per_out = []

for idx in range(len(dat_out_ON)):
    pr = model_train.predict(dat_out_ON[idx])

    per_out.append(metrics.balanced_accuracy_score(label_out_ON[idx], pr))

plt.figure(figsize=(4, 3), dpi=300)
plt.bar(np.arange(len(per_out)), per_out)
plt.xlabel("Test subject number")
plt.ylim(0.5, 0.75)
plt.axhline(0.73, label="Patient Ind. OFF->ON", color="gray")
plt.ylabel("Balanced accuracy")
plt.title("All Stim-OFF data predict Stim-ON")
plt.legend()
plt.tight_layout()

