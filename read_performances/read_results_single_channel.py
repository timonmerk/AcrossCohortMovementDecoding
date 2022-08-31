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

    READ_ALL = True

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
    df_best = []
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
            performance_dict={}, read_mov_detection_rates=True
        )
        df_per = feature_reader.get_dataframe_performances(per)

        df_per["cohort"] = current_cohort

        df_per["run"] = RUN_NAME
        if current_cohort == "Washington":
            df_per["sub"] = RUN_NAME[:2]
        df_all.append(df_per)

        df_ind_ch = df_per.query("ch_type == 'electrode ch'").reset_index()
        idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
        df_best_sub_ind_ch = df_ind_ch.iloc[idx]
        df_best.append(df_best_sub_ind_ch)

        df_ind_ch = df_per.query("ch_type == 'cortex grid'").reset_index()
        idx = df_ind_ch.groupby(["sub"])["InnerCV_performance_test"].idxmax()
        df_best_sub_ind_ch = df_ind_ch.iloc[idx]
        df_best.append(df_best_sub_ind_ch)

    df_comb = pd.concat(df_all)
    df_comb["x"] = df_comb["coord"].apply(lambda x: x[0])
    df_comb["y"] = df_comb["coord"].apply(lambda x: x[1])
    df_comb["z"] = df_comb["coord"].apply(lambda x: x[2])
    df_mean_runs = (
        df_comb.groupby(["sub", "cohort", "ch_type", "ch"]).mean().reset_index()
    )

    df_ch = df_mean_runs.query('ch_type == "electrode ch"')[
        [
            "sub",
            "cohort",
            "ch",
            "performance_test",
            "mov_detection_rates_test",
            "x",
            "y",
            "z",
        ]
    ]
    df_ch.to_csv("df_ch_performances.csv")

    df_gp = df_mean_runs.query('ch_type == "electrode ch"')[
        [
            "sub",
            "cohort",
            "ch",
            "performance_test",
            "mov_detection_rates_test",
            "x",
            "y",
            "z",
        ]
    ]
    df_gp.to_csv("df_grid_point_performances.csv")

    pd.concat(df_best).to_pickle("df_best.p")

    # df_gp = (
    #    df_mean_runs.query('ch_type == "cortex grid"')
    #    .groupby(["ch"])
    #    .mean()
    #    .reset_index()[["sub", "ch", "performance_test", "mov_detection_rates_test", "x", "y", "z"]]
    # )
    # df_gp.to_csv("df_grid_point_performances.csv")

    nm_plots.plot_df_subjects(
        df=df_comb.groupby(["sub", "cohort", "ch_type"]).mean().reset_index(),
        x_col="cohort",
        y_col="performance_test",
        hue="ch_type",
    )
