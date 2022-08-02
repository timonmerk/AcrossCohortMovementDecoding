import os
import numpy as np
import pandas as pd
from py_neuromodulation import nm_RMAP

# get single channel performances

# load fingerprints str. and func.

rmap = nm_RMAP.RMAPChannelSelector()

df = pd.read_csv(os.path.join("read_performances", "df_ch_performances.csv"))

path_connectomes = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes"

cohorts = ["Washington", "Berlin", "Beijing", "Pittsburgh"]  #

connectivity_metric = "structural connectivity"
l_names = []
l_dat = []

for cohort in cohorts:
    if connectivity_metric == "structural connectivity":
        l_fps_names, l_fps_dat = rmap.load_all_fingerprints(
            os.path.join(path_connectomes, cohort, connectivity_metric),
            cond_str=None,  # "_AvgR_Fz.nii"
        )
    else:
        l_fps_names, l_fps_dat = rmap.load_all_fingerprints(
            os.path.join(path_connectomes, cohort, connectivity_metric),
            cond_str="_AvgR_Fz.nii",  # "_AvgR_Fz.nii"
        )
    l_names.append(l_fps_names)
    l_dat.append(l_fps_dat)

l_names = np.concatenate(l_names)
l_dat = np.concatenate(l_dat)

l_per = []
dat_arr = []
for idx, f in enumerate(l_names):
    cohort = f.split("_")[0]
    sub = f.split("_")[1]
    if (sub == "013" or sub == "014") and cohort == "Berlin":
        continue  # this was previously not estimated

    if connectivity_metric == "functional connectivity":
        ch = f[f.find("ECOG") : f.find("_func")]
    else:
        ch = f[f.find("ECOG") : f.find("_struc")]
        cohort = cohort[1:]

    try:
        l_per.append(
            df.query("cohort == @cohort and ch.str.contains(@ch) and sub == @sub").iloc[
                0
            ]["performance_test"]
        )

        dat_arr.append(np.nan_to_num(l_dat[idx].flatten()))
    except:
        print(f"{cohort} {sub} {ch} failed")

per_arr = np.array(l_per)
per_arr[np.isnan(per_arr)] = 0.5

rmap_arr_np = np.nan_to_num(rmap.get_RMAP(np.array(dat_arr).T, per_arr))
if connectivity_metric == "functional connectivity":
    rmap.save_Nii(rmap_arr_np, rmap.affine, name="rmap_func.nii")
else:
    rmap.save_Nii(rmap_arr_np, rmap.affine, name="rmap_struc.nii")
