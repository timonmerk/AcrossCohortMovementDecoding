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

if __name__ == "__main__":

    PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"
    files_all = []
    cohorts = ["Pittsburgh", "Beijing", "Washington", "Berlin"]
    for cohort in cohorts:
        files_all.append(
            [
                os.path.join(PATH_OUT, cohort, f)
                for f in os.listdir(os.path.join(PATH_OUT, cohort))
            ]
        )
    files_all = np.concatenate(files_all)

    df_all = []

    for PATH_OUT_RUN in files_all:
        for cohort in cohorts:
            if cohort in PATH_OUT_RUN:
                current_cohort = cohort
        RUN_NAME = os.path.basename(PATH_OUT_RUN)
        print(RUN_NAME)
        feature_reader = nm_analysis.Feature_Reader(
            feature_dir=os.path.join(PATH_OUT, current_cohort), feature_file=RUN_NAME
        )

        per = feature_reader.read_results(
            performance_dict={},
            ML_model_name="LM_all_comb",
            read_grid_points=False,
            read_channels=False,
            read_all_combined=True,
        )
        df_per = feature_reader.get_dataframe_performances(per)

        df_per["cohort"] = current_cohort

        df_per["run"] = RUN_NAME
        if current_cohort == "Washington":
            df_per["sub"] = RUN_NAME[:2]
        df_all.append(df_per)
    df = pd.concat(df_all)
