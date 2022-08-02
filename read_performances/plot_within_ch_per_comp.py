import os
import pandas as pd

from py_neuromodulation import nm_plots


if __name__ == "__main__":

    df_best = pd.read_pickle("read_performances\\df_best.p")
    df_best = (
        df_best.groupby(["cohort", "sub", "ch_type"]).mean().reset_index()
    )  # mean over runs!
    df_best.drop(df_best.filter(regex="index"), axis=1, inplace=True)

    df_all_ch = pd.read_csv("read_performances\\df_all_comb_performances.csv")
    df_plt_all_ch = df_all_ch.groupby(["sub", "cohort"]).mean().reset_index()
    df_plt_all_ch.drop(df_plt_all_ch.filter(regex="Unname"), axis=1, inplace=True)
    df_plt_all_ch["ch_type"] = "all combined"

    # read here also the df_rmap
    df_rmap = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")
    df_rmap["ch_type"] = "rmap_select"
    df_rmap = df_rmap[["sub", "cohort", "performance_test", "ch_type"]].reset_index()

    df_plt_all = pd.concat(
        [
            df_best[["sub", "cohort", "performance_test", "ch_type"]],
            df_plt_all_ch[["sub", "cohort", "performance_test", "ch_type"]],
            df_rmap,
        ]
    )

    df_plt_all = df_plt_all.rename(
        columns={
            "cohort": "Cohort",
            "performance_test": "Balanced Accuracy",
            "ch_type": "Channel Type",
        }
    )
    nm_plots.plot_df_subjects(
        df=df_plt_all,
        x_col="Cohort",
        y_col="Balanced Accuracy",
        hue="Channel Type",
        title="Individual subject performances",
        PATH_SAVE=os.path.join("figure", "ind_sub_comp_with_rmap.pdf"),
    )
