from py_neuromodulation import nm_analysis
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, model_selection, metrics, discriminant_analysis
import pandas as pd

PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_no_kf\Berlin"

run_names = os.listdir(PATH_OUT)

labels_ = []
data_ = []
ch_add_l = []

label = "SQUARED_EMG"

for idx, run_name in enumerate(run_names):

    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=run_name
    )

    if feature_reader.nm_channels.query("name.str.contains('ECOG_R')").shape[0] > 0:
        # use right STN channels
        chs_add = feature_reader.nm_channels.query("type == 'dbs' and name.str.contains('_R_')")["new_name"]
    elif feature_reader.nm_channels.query("name.str.contains('ECOG_L')").shape[0] > 0:
        # use left STN channels
        chs_add = feature_reader.nm_channels.query("type == 'dbs' and name.str.contains('_L_')")["new_name"]

    if label not in list(feature_reader.feature_arr.columns):
        label_try = 'SQUARED_INTERPOLATED_EMG'
        if label_try in list(feature_reader.feature_arr.columns):
            labels_.append(feature_reader.feature_arr[label_try])
        else:
            print("nope")
            continue
    else:
        labels_.append(feature_reader.feature_arr[label])
    ch_add_l.append(chs_add)
    cols_ = [col for ch in chs_add for col in list(feature_reader.feature_arr.columns) if ch in col]
    data_.append(feature_reader.feature_arr[cols_])

data_best = []
label_add = []
scores_outer = []
for run_idx in range(len(ch_add_l)):

    scores_run_l = []
    for ch in list(ch_add_l[run_idx]):
        df_ch = data_[run_idx][[c for c in data_[run_idx].columns if ch in c]]
        if df_ch.shape[1] == 0:
            print("no ch data")
            continue
        y = labels_[run_idx]

        model = linear_model.LogisticRegression()

        y_ = y != 0
        pr = model_selection.cross_val_predict(
            estimator=model,
            X=df_ch,
            y=y_
        )
        score_ = metrics.balanced_accuracy_score(y_, pr)
        scores_run_l.append(score_)
    best_ch_idx = np.argmax(scores_run_l)
    best_ch = list(ch_add_l[run_idx])[best_ch_idx]
    data_best_to_add = data_[run_idx][[c for c in data_[run_idx].columns if best_ch in c]]

    data_best.append(data_best_to_add)
    label_add.append(y)
    scores_outer.append(scores_run_l[best_ch_idx])


# check only performances above 0.6 ba

# select best channel, train model and save
idx_select = np.argmax(np.array(scores_outer))
X = data_best[idx_select]
y = label_add[idx_select]
model = linear_model.LogisticRegression()
pr = model_selection.cross_val_predict(
    estimator=model,
    X=X,
    y=y
)
print(metrics.balanced_accuracy_score(y, pr))
model = linear_model.LogisticRegression()
model.fit(X, y)
import pickle
pickle.dump(model, open("best_model_STN_best_ch.p", 'wb'))


idx_select = np.where(np.array(scores_outer) > 0.6)[0]

dat_ = []
y_all = []
for idx in idx_select:
    dat_.append(data_best[idx])
    y_all.append(label_add[idx] != 0)
X = np.concatenate(dat_)
y = np.concatenate(y_all)

model = linear_model.LogisticRegression()

model = discriminant_analysis.LinearDiscriminantAnalysis()

pr = model_selection.cross_val_predict(
    estimator=model,
    X=X,
    y=y
)
print(metrics.balanced_accuracy_score(y, pr))

model = linear_model.LogisticRegression()
model.fit(X, y)

# save model

import pickle
pickle.dump(model, open("trained_model_STN_best_ch.p", 'wb'))


########################################### CHECK HERE test for Analog Rotameter
plt.plot(np.concatenate(labels_))



list_all_features_STN = []
labels_ = []
data = []
dat_to_add_l = []
for idx, run_name in enumerate(run_names):

    label = "ANALOG_R_ROTA_CH"

    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=run_name
    )

    if label not in list(feature_reader.feature_arr.columns):
        #print(feature_reader.feature_arr.columns)
        label = "ANALOG_L_ROTA_CH" 
        if label not in list(feature_reader.feature_arr.columns):
            print("NICHT DA")
            continue
    scaler = MinMaxScaler()

    dat_to_add = np.abs(
        np.diff(
            feature_reader.feature_arr[label]
        )
    )
    h_percentile = np.percentile(dat_to_add, 90)

    dat_to_add_clipped = np.clip(
        dat_to_add,
        a_min=np.percentile(dat_to_add, 10),
        a_max=h_percentile
    )

    #plt.plot(np.abs(np.diff(
    #    feature_reader.feature_arr[label]
    #)))

    label_val = scaler.fit_transform(
        np.expand_dims(
            dat_to_add_clipped,
            axis=1
        )
    )
    labels_.append(label_val[:, 0])
    dat_to_add_l.append(dat_to_add)
    #plt.figure()
    #plt.plot(label_val[:, 0])
    #plt.show()

    data.append(feature_reader.feature_arr)
    # set label to be diff of analog and abs value

    #feature_reader.feature_arr
    # get best channel pair per patient

    # concantenate features

    # plt.plot(feature_reader.feature_arr["time"][1:]/1000, label_val)

print("ha")


for i in range(len(labels_)):
    plt.plot(labels_[0])
    plt.plot(dat_to_add_l[0])
    plt.show()

plt.show()

plt.plot(np.concatenate(labels_))

plt.show()

plt.plot(feature_reader.feature_arr["time"][1:]/1000, label_val)
plt.title(label)
plt.ylabel("a.u.")
plt.xlabel("Time [s]")