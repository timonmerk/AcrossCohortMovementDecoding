# read stim off runs and then read data from stim on

import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats, nm_across_patient_decoding, nm_decode, nm_plots

RUN_PARRM_STIM_DATA = False

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


def get_data_stim(ap_runner, sub: str, ch: str, stim_on: bool = True):
    X_train = []
    y_train = []
    for f in ap_runner.ch_all["Berlin"][sub][ch].keys():
        if stim_on is True and "StimOn" in f:
            if RUN_PARRM_STIM_DATA is True:
                X_train.append(ap_runner_parrm.ch_all["Berlin"][sub][ch][f]["data"])
                y_train.append(ap_runner_parrm.ch_all["Berlin"][sub][ch][f]["label"])
            else:
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
        X_test, y_test = get_data_stim(ap_runner_parrm, sub, ch, stim_on=True)
    else:
        X_test, y_test = get_data_stim(ap_runner, sub, ch, stim_on=True)
    X_train, y_train = get_data_stim(ap_runner, sub, ch, stim_on=False)

    decoder = ap_runner.init_decoder()
    model = decoder.wrapper_model_train(
        X_train=X_train,
        y_train=y_train,
        return_fitted_model_only=True,
    )
    cv_res = decoder.eval_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
        save_data=False,
    )
    # read individual performance
    # plot cross predicted performance

    # ok, there are multiple runs associated with that

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": per_ON,
                    "Test Performance Detection Rate": cv_res.mov_detection_rates_test[
                        0
                    ],
                    "Subject": sub,
                    "Model Type": "STIM ON",
                },
                index=[0],
            ),
        ]
    )

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": per_OFF,
                    "Test Performance Detection Rate": cv_res.mov_detection_rates_test[
                        0
                    ],
                    "Subject": sub,
                    "Model Type": "STIM OFF",
                },
                index=[0],
            ),
        ]
    )

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": cv_res.score_test[0],
                    "Test Performance Detection Rate": cv_res.mov_detection_rates_test[
                        0
                    ],
                    "Subject": sub,
                    "Model Type": "STIM OFF->ON Predict",
                },
                index=[0],
            ),
        ]
    )

    if RUN_PARRM_STIM_DATA is True:
        X_train, y_train = get_data_stim(ap_runner_parrm, sub, ch, stim_on=True)
    else:
        X_train, y_train = get_data_stim(ap_runner, sub, ch, stim_on=True)
    X_test, y_test = get_data_stim(ap_runner, sub, ch, stim_on=False)

    decoder = ap_runner.init_decoder()
    model = decoder.wrapper_model_train(
        X_train=X_train,
        y_train=y_train,
        return_fitted_model_only=True,
    )
    cv_res = decoder.eval_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        cv_res=nm_decode.CV_res(get_movement_detection_rate=True),
        save_data=False,
    )

    df_comp_STIM = pd.concat(
        [
            df_comp_STIM,
            pd.DataFrame(
                {
                    "Test Performance": cv_res.score_test[0],
                    "Test Performance Detection Rate": cv_res.mov_detection_rates_test[
                        0
                    ],
                    "Subject": sub,
                    "Model Type": "STIM ON->OFF Predict",
                },
                index=[0],
            ),
        ]
    )

if RUN_PARRM_STIM_DATA is True:
    df_comp_STIM.to_csv("df_STIM_ON_OFF_parrm.csv")
else:
    df_comp_STIM.to_csv("df_STIM_ON_OFF.csv")

# add here also a combined model that used stim off and on as training and testing
# but report the performances separate for both classes
# add also STIM ON --> OFF


nm_plots.plot_df_subjects(
    df=df_comp_STIM,
    x_col="Model Type",
    y_col="Test Performance Detection Rate",  # Test Performance
    title="Berlin Stim On/Off Comparison",
    hue=None,
    figsize_tuple=(4, 5)
    # PATH_SAVE=os.path.join("figure", f"stim_off_on_comp_predict.pdf"),
)
plt.ylim(0.5, 1.05)
plt.ylabel("Movement detection rate")  # Balanced accuracy
plt.tight_layout()


plt.figure(figsize=(6, 5), dpi=300)
sb.barplot(
    data=df_comp_STIM,
    x="Subject",
    y="Test Performance",
    hue="Model Type",
    palette="viridis",
)
plt.legend(bbox_to_anchor=(1.04, 1))
plt.title(f"STIM ON / OFF Comparison Berlin subjects")
plt.savefig(
    os.path.join("figure", "stim_off_on_comp_predict.pdf"),
    bbox_inches="tight",
)
