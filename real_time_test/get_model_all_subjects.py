import numpy as np
import os
import pandas as pd
import pickle
from sklearn import linear_model, metrics

@staticmethod
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
    X_test, y_test = get_data_sub_ch(
        channel_all, cohort_test, sub_test, ch_test
    )
    return X_test, y_test

outpath = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"
df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

channel_all = np.load(
    os.path.join(outpath, "channel_all.npy"),
    allow_pickle="TRUE",
).item()

X_train_comb = []
y_train_comb = []
for cohort_train in list(channel_all.keys()):
    for sub_train in channel_all[cohort_train]:

        X_train, y_train = get_data_channels(
            sub_train, cohort_train, df_rmap=df_rmap_corr
        )

        X_train_comb.append(X_train)
        y_train_comb.append(y_train)
if len(X_train_comb) > 1:
    X_train = np.concatenate(X_train_comb, axis=0)
    y_train = np.concatenate(y_train_comb, axis=0)
else:
    X_train = X_train_comb[0]
    y_train = X_train_comb[0]

model = linear_model.LogisticRegression(class_weight="balanced").fit(X_train, y_train)

print(metrics.balanced_accuracy_score(y_train, model.predict(X_train)))
with open('real_time_test\\model_all_subs_classweights_balanced.pkl', 'wb') as f:
    pickle.dump(model, f)


model = linear_model.LogisticRegression().fit(X_train, y_train)

print(metrics.balanced_accuracy_score(y_train, model.predict(X_train)))
with open('real_time_test\\model_all_subs_classweights_non_balanced.pkl', 'wb') as f:
    pickle.dump(model, f)


IDX = 41
print(metrics.balanced_accuracy_score(y_train_comb[IDX], linear_model.LogisticRegression().fit(X_train_comb[IDX], y_train_comb[IDX]).predict(X_train_comb[IDX])))
