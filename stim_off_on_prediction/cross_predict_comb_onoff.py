# read stim off runs and then read data from stim on

import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats, nm_across_patient_decoding, nm_decode, nm_plots

RUN_PARRM_STIM_DATA = False
RUN_BANDSTOP_FILTER_EST = True

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

# with open(os.path.join("read_performances", "df_all.p"), "rb") as handle:
#    df_all = pickle.load(handle)

df_all = pd.read_pickle(os.path.join("read_performances", "df_all.p"))

df = df_all.query("ch_type == 'electrode ch' and cohort != 'Washington'")

# some patients have stim on and off runs
df_STIMON = df.query("run.str.contains('StimOn')")
df_STIMOFF = df.query("run.str.contains('StimOff') and cohort == 'Berlin'")

ap_runner = nm_across_patient_decoding.AcrossPatientRunner(
    outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft",
    cohorts=["Beijing", "Pittsburgh", "Berlin"],
    use_nested_cv=False,
    ML_model_name="LMGridPoints",
    load_channel_all=True,
)

ap_runner_parrm = nm_across_patient_decoding.AcrossPatientRunner(
    outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_parrm_removed",
    cohorts=["Berlin"],
    use_nested_cv=False,
    load_channel_all=True,
)

ap_runner_bandstop = nm_across_patient_decoding.AcrossPatientRunner(
    outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_bandstop_filtered",
    cohorts=["Berlin"],
    use_nested_cv=False,
    load_channel_all=True,
)


def get_data_stim(ap_runner, stim_on: bool = True):
    X_train = []
    y_train = []
    for f in ap_runner.ch_all["Berlin"][sub][ch].keys():
        if stim_on is True and "StimOn" in f:
            X_train.append(ap_runner.ch_all["Berlin"][sub][ch][f]["data"])
            y_train.append(ap_runner.ch_all["Berlin"][sub][ch][f]["label"])
        if stim_on is False and "StimOff" in f:
            X_train.append(ap_runner.ch_all["Berlin"][sub][ch][f]["data"])
            y_train.append(ap_runner.ch_all["Berlin"][sub][ch][f]["label"])

    if len(X_train) > 1:
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
    else:
        X_train = X_train[0]
        y_train = y_train[0]
    return X_train, y_train


# for each subject select best RMAP channel
df_comp_STIM = pd.DataFrame()
for sub in df_STIMON["sub"].unique():
    ch = df_rmap_corr.query("cohort=='Berlin' and sub == @sub").iloc[0].ch
    per_ON = df_STIMON.query("sub == @sub and ch == @ch")["performance_test"].mean()
    per_OFF = df_STIMOFF.query("sub == @sub and ch == @ch")["performance_test"].mean()

    if RUN_PARRM_STIM_DATA is True:
        X_on, y_on = get_data_stim(ap_runner_parrm, stim_on=True)
    elif RUN_BANDSTOP_FILTER_EST is True:
        X_on, y_on = get_data_stim(ap_runner_bandstop, stim_on=True)
    else:
        X_on, y_on = get_data_stim(ap_runner, stim_on=True)

    X_off, y_off = get_data_stim(ap_runner, stim_on=False)

    idx_shuffle = np.random.permutation(y_on.shape[0])
    idx_train = int(0.66 * y_on.shape[0])
    X_train_on = X_on[idx_shuffle][:idx_train]
    y_train_on = y_on[idx_shuffle][:idx_train]
    X_test_on = X_on[idx_shuffle][idx_train:]
    y_test_on = y_on[idx_shuffle][idx_train:]

    idx_shuffle = np.random.permutation(y_off.shape[0])
    idx_train = int(0.66 * y_off.shape[0])
    X_train_off = X_off[idx_shuffle][:idx_train]
    y_train_off = y_off[idx_shuffle][:idx_train]
    X_test_off = X_off[idx_shuffle][idx_train:]
    y_test_off = y_off[idx_shuffle][idx_train:]

    X_train = np.concatenate((X_train_off, X_train_on), axis=0)
    y_train = np.concatenate((y_train_off, y_train_on))

    decoder = ap_runner.init_decoder()
    model = decoder.wrapper_model_train(
        X_train=X_train,
        y_train=y_train,
        return_fitted_model_only=True,
    )
    cv_res = decoder.eval_model(
        model,
        X_train,
        X_test_on,
        y_train,
        y_test_on,
        cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
        save_data=False,
    )

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": cv_res.score_test[0],
                    "Subject": sub,
                    "Model Type": "STIM ON-OFF->ON Predict",
                },
                index=[0],
            ),
        ]
    )

    cv_res = decoder.eval_model(
        model,
        X_train,
        X_test_off,
        y_train,
        y_test_off,
        cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
        save_data=False,
    )

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": cv_res.score_test[0],
                    "Subject": sub,
                    "Model Type": "STIM ON-OFF->OFF Predict",
                },
                index=[0],
            ),
        ]
    )

if RUN_PARRM_STIM_DATA is True:
    df_comp_STIM.to_csv(
        os.path.join("stim_off_on_prediction", "df_STIM_ON_OFF_predict_parrm.csv")
    )
elif RUN_BANDSTOP_FILTER_EST is True:
    df_comp_STIM.to_csv(
        os.path.join("stim_off_on_prediction", "df_STIM_ON_OFF_predict_bandstop.csv")
    )
else:
    df_comp_STIM.to_csv(
        os.path.join("stim_off_on_prediction", "df_STIM_ON_OFF_predict.csv")
    )
