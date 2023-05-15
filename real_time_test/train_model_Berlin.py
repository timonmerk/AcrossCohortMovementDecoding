from py_neuromodulation import nm_analysis
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model, model_selection, metrics, discriminant_analysis
import pandas as pd
import pickle

PATH_FEATURES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_no_kf\Berlin"

run_names = os.listdir(PATH_FEATURES)



label = "SQUARED_EMG"
ch_type = "ecog"  # "dbs"
MED_OFF = True
STIM_OFF = True
PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\real_time_test\model_output"


def get_data_all(run_names: list, MED_OFF: bool, STIM_OFF: bool, ch_type: str):
    runs_use = []
    labels_ = []
    data_ = []
    ch_add_l = []
    for idx, run_name in enumerate(run_names):

        if MED_OFF is True and "MedOff" not in run_name:
            continue
        elif MED_OFF is False and "MedOn" not in run_name:
            continue
        if STIM_OFF is True and "StimOff" not in run_name:
            continue
        elif STIM_OFF is False and "StimOn" not in run_name:
            continue

        runs_use.append(run_name)
        feature_reader = nm_analysis.Feature_Reader(
            feature_dir=PATH_FEATURES, feature_file=run_name
        )

        if feature_reader.nm_channels.query("name.str.contains('ECOG_R')").shape[0] > 0:
            # use right STN channels
            chs_add = feature_reader.nm_channels.query("type == @ch_type and name.str.contains('_R_') and status == 'good'")["new_name"]
        elif feature_reader.nm_channels.query("name.str.contains('ECOG_L')").shape[0] > 0:
            # use left STN channels
            chs_add = feature_reader.nm_channels.query("type == @ch_type and name.str.contains('_L_') and status == 'good'")["new_name"]

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
    return data_, labels_, ch_add_l, cols_, runs_use

def get_data_best_channel_and_performances(data_: list, labels_: list, ch_add_l: list):
    data_best = []
    label_add = []
    scores_outer = []
    df_per = pd.DataFrame()

    for run_idx in range(len(ch_add_l)):

        scores_run_l = []
        for ch in list(ch_add_l[run_idx]):
            df_ch = data_[run_idx][[c for c in data_[run_idx].columns if ch in c]]
            if df_ch.shape[1] == 0:
                print("no ch data")  # since e.g. ECOG_L_1_SMC_AT was not included
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

            df_per = df_per.append({
                "sub" : runs_use[run_idx][:7],
                "run" : runs_use[run_idx],
                "ch" : ch,
                "per" : score_,
            }, ignore_index=True)

            scores_run_l.append(score_)
        best_ch_idx = np.argmax(scores_run_l)
        best_ch = list(ch_add_l[run_idx])[best_ch_idx]
        data_best_to_add = data_[run_idx][[c for c in data_[run_idx].columns if best_ch in c]]

        data_best.append(data_best_to_add)
        label_add.append(y)
        scores_outer.append(scores_run_l[best_ch_idx])
    return data_best, label_add, scores_outer, df_per

def get_model_best_channel_overall(scores_outer: list, data_best: list, label_add: list, save_path: str="model_ECoG_best_ch.p"):
    # check only performances above 0.6 ba
    # select best channel ALL, train model and save
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
    if save_path is not None:
        pickle.dump(model, open(save_path, 'wb'))
    return model

def get_all_models_above_thr(scores_outer: list, data_best: list, label_add: list, save_path: str="model_ECoG_best_channels.p"):

    idx_select = np.where(np.array(scores_outer) > 0.6)[0]

    dat_ = []
    y_all = []
    for idx in idx_select:
        dat_.append(data_best[idx])
        y_all.append(label_add[idx] != 0)
    X = np.concatenate(dat_)
    y = np.concatenate(y_all)

    model = linear_model.LogisticRegression()

    #model = discriminant_analysis.LinearDiscriminantAnalysis()

    pr = model_selection.cross_val_predict(
        estimator=model,
        X=X,
        y=y
    )
    print(metrics.balanced_accuracy_score(y, pr))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    if save_path is not None:
        pickle.dump(model, open(save_path, 'wb'))

    return model


SAVE_PER = True
for ch_type in ["ecog", "dbs"]:
    for MED_OFF_STATE in [False, True]:

        str_add = f"{ch_type}_MedOff" if MED_OFF_STATE is True else f"{ch_type}_MedOn"
        data_, labels_, ch_add_l, cols_, runs_use = get_data_all(run_names, MED_OFF_STATE, STIM_OFF=True, ch_type=ch_type)
        data_best, label_add, scores_outer, df_per = get_data_best_channel_and_performances(data_, labels_, ch_add_l)
        

        if SAVE_PER is True:
            df_per.to_csv(os.path.join(PATH_OUT, f"per_{str_add}.csv"))
        get_model_best_channel_overall(scores_outer, data_best, label_add, save_path=os.path.join(PATH_OUT, f"best_single_ch_{str_add}.p"))

        get_all_models_above_thr(scores_outer, data_best, label_add, save_path=os.path.join(PATH_OUT, f"best_channels_{str_add}.p"))


## compute RMAP 

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