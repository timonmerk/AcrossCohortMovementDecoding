import os
import mne_bids
import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from bids import BIDSLayout
from py_neuromodulation import nm_across_patient_decoding, nm_analysis, nm_decode
from sklearn import linear_model, metrics
from multiprocessing import Pool


def run_decoding(PATH_OUT_RUN):
    RUN_NAME = os.path.basename(PATH_OUT_RUN)
    PATH_OUT = os.path.dirname(PATH_OUT_RUN)
    feature_reader = nm_analysis.Feature_Reader(
        feature_dir=PATH_OUT, feature_file=RUN_NAME
    )

    # need to check which one is selected for Berlin and Beijing

    feature_reader.decoder = nm_decode.Decoder(
        features=feature_reader.feature_arr,
        label=feature_reader.label,
        label_name=feature_reader.label_name,
        used_chs=feature_reader.used_chs,
        model=linear_model.LogisticRegression(
            class_weight="balanced",
        ),
        eval_method=metrics.balanced_accuracy_score,
        cv_method="NonShuffledTrainTestSplit",
        get_movement_detection_rate=True,
        min_consequent_count=3,
        TRAIN_VAL_SPLIT=False,
        RUN_BAY_OPT=False,
        bay_opt_param_space=None,
        use_nested_cv=True,
        sfreq=feature_reader.settings["sampling_rate_features_hz"],
    )

    performances = feature_reader.run_ML_model(
        estimate_channels=True,
        estimate_gridpoints=False,
        estimate_all_channels_combined=False,
        save_results=True,
        output_name="LM",
    )
    return performances


if __name__ == "__main__":
    PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_parrm_removed"
    PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_bandstop_filtered"
    files_all = []

    subs = ["002", "005", "006", "007", "008", "009"]

    for cohort in ["Berlin"]:
        files_all.append(
            [
                os.path.join(PATH_OUT, cohort, f)
                for f in os.listdir(os.path.join(PATH_OUT, cohort))
            ]
        )
    files_all = np.concatenate(files_all)
    df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

    for f in files_all:
        performances = run_decoding(
            f,
        )
        sub = f[f.find("sub-") + 4 : f.find("sub-") + 7]
        ch = df_rmap_corr.query("cohort=='Berlin' and sub == @sub").iloc[0].ch
        print(sub)
        print(f"ba: {performances[sub][ch]['performance_test']}")
        print(f"mdr: {performances[sub][ch]['mov_detection_rates_test']}")

    # pool = Pool(processes=50)
    # pool.map(run_decoding, files_all)
