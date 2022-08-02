import os
import numpy as np
import pandas as pd
from py_neuromodulation import nm_RMAP


# 1. load all fingerprints -> functional connectivity results

rmap = nm_RMAP.RMAPChannelSelector()

df = pd.read_csv(os.path.join("read_performances", "df_ch_performances.csv"))

path_connectomes = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes"

cohorts = ["Washington", "Berlin", "Beijing", "Pittsburgh"]

connectivity_metric = "functional connectivity"
l_names = []
l_dat = []

for cohort in cohorts:
    l_fps_names, l_fps_dat = rmap.load_all_fingerprints(
        os.path.join(path_connectomes, cohort, connectivity_metric),
        cond_str="_AvgR_Fz.nii",  #
    )
    l_names.append(l_fps_names)
    l_dat.append(l_fps_dat)

l_names = np.concatenate(l_names)
l_dat = np.concatenate(l_dat)

# load RMAP
rmap_func = rmap.load_fingerprint("across patient running\\RMAP\\rmap_func.nii")
rmap_func = rmap_func.flatten()

rmap_corr = []
for idx, _ in enumerate(l_dat):
    rmap_corr.append(rmap.get_corr_numba(rmap_func, l_dat[idx].flatten()))

# insert correlation values into df
df["r_func"] = -1

vals_r = []
for idx, f in enumerate(l_names):
    cohort = f.split("_")[0]
    sub = f.split("_")[1]
    if connectivity_metric == "functional connectivity":
        ch = f[f.find("ECOG") : f.find("_func")]
    else:
        ch = f[f.find("ECOG") : f.find("_struc")]
        cohort = cohort[1:]
    try:
        row_idx = (
            df.query("cohort == @cohort and ch.str.contains(@ch) and sub == @sub")
            .iloc[0]
            .name
        )
        df.at[row_idx, "r_func"] = rmap_corr[idx]
    except:
        continue

df.to_csv(os.path.join("read_performances", "df_ch_performances.csv"))

idx_max_r = df.groupby(["sub", "cohort"])["r_func"].transform(max) == df["r_func"]

df_best_r = df[idx_max_r]

df_best_r.to_csv("df_best_func_rmap_ch.csv")

# 1. step plot the performance comparison
