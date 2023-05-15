import os
import pandas as pd
import numpy as np

from py_neuromodulation import nm_plots


if __name__ == "__main__":

    # plot movement detection rate

    df_best = pd.read_pickle("read_performances\\df_best.p")
    df_best = (
        df_best.groupby(["cohort", "sub", "ch_type"]).mean().reset_index()
    )  # mean over runs!
    df_best.drop(df_best.filter(regex="index"), axis=1, inplace=True)

    # read here df_ch_performances.csv instead
    df_ch = pd.read_csv("read_performances\\df_ch_performances.csv")
    df_ch = df_ch.groupby(["cohort", "sub"]).max().reset_index()  # best channel
    df_ch["ch_type"] = "Individual channels"
    df_ch = df_ch[["sub", "cohort", "performance_test", "mov_detection_rates_test", "ch_type"]]

    # read grid point performances
    df_gp = pd.read_csv("read_performances\\df_grid_point_performances.csv")
    df_gp = df_gp.groupby(["cohort", "sub"]).max().reset_index()
    df_gp["ch_type"] = "Grid points"
    df_gp = df_gp[["sub", "cohort", "performance_test", "mov_detection_rates_test", "ch_type"]]

    df_all_ch = pd.read_csv("read_performances\\df_all_comb_performances.csv")
    df_plt_all_ch = df_all_ch.groupby(["sub", "cohort"]).mean().reset_index()
    df_plt_all_ch.drop(df_plt_all_ch.filter(regex="Unname"), axis=1, inplace=True)
    df_plt_all_ch["ch_type"] = "all combined"

    # read here also the df_rmap
    df_rmap = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")
    df_rmap["ch_type"] = "rmap_select"
    df_rmap = df_rmap[
        ["sub", "cohort", "performance_test", "mov_detection_rates_test", "ch_type"]
    ].reset_index()

    df_plt_all = pd.concat(
        [
            df_ch,
            df_gp,
            df_plt_all_ch[
                [
                    "sub",
                    "cohort",
                    "performance_test",
                    "mov_detection_rates_test",
                    "ch_type",
                ]
            ],
            df_rmap,
        ]
    )

    df_plt_all = df_plt_all.rename(
        columns={
            "cohort": "Cohort",
            "performance_test": "Balanced Accuracy",
            "mov_detection_rates_test": "Movement Detection Rate",
            "ch_type": "Channel Type",
        }
    )
    nm_plots.plot_df_subjects(
        df=df_plt_all[np.logical_or(df_plt_all["Channel Type"].str.contains("all combined"), df_plt_all["Channel Type"].str.contains("Individual"))] ,
        x_col="Cohort",
        y_col="Balanced Accuracy",
        hue="Channel Type",
        title="Individual subject performances",
        PATH_SAVE=os.path.join("figure", "ind_sub_comp_with_rmap_v3.pdf"),
    )

    nm_plots.plot_df_subjects(
        df=df_plt_all[np.logical_or(df_plt_all["Channel Type"].str.contains("all combined"), df_plt_all["Channel Type"].str.contains("Individual"))] ,
        x_col="Cohort",
        y_col="Movement Detection Rate",
        hue="Channel Type",
        title="Individual subject performances",
        PATH_SAVE=os.path.join("figure", "ind_sub_comp_with_rmap_mov_det_v3.pdf"),
    )
