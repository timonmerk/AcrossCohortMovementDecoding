import os
import mne_bids
import mne
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

    if "Washington" in PATH_OUT_RUN:
        # check which label is being used
        # change for Washington
        mov_starts = np.where(np.diff(feature_reader.feature_arr["mov"]) > 0)[0]
        seg_cut = []
        for mov_start in mov_starts:
            for i in range(5):
                seg_cut.append(mov_start + i)

        ind_cut = np.concatenate(
            (np.where(feature_reader.feature_arr["mov"] == 11)[0], seg_cut)
        )
        idx_select = set(np.arange(feature_reader.feature_arr["mov"].shape[0])) - set(
            ind_cut
        )
        feature_reader.feature_arr = feature_reader.feature_arr.iloc[
            list(idx_select), :
        ].reset_index(drop=True)
        # analyzer.feature_arr["mov"] = analyzer.feature_arr["mov"] > 0
        feature_reader.label = np.array(
            feature_reader.feature_arr["mov"] > 0, dtype=int
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
        estimate_channels=False,
        estimate_gridpoints=False,
        estimate_all_channels_combined=True,
        save_results=True,
        output_name="LM_all_comb",
    )


if __name__ == "__main__":

    PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"
    files_all = []
    for cohort in ["Pittsburgh", "Beijing", "Washington", "Berlin"]:
        files_all.append(
            [
                os.path.join(PATH_OUT, cohort, f)
                for f in os.listdir(os.path.join(PATH_OUT, cohort))
            ]
        )
    files_all = np.concatenate(files_all)
    # run_decoding(files_all[100])

    pool = Pool(processes=50)
    pool.map(run_decoding, files_all)
