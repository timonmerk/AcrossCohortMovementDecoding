from py_neuromodulation import nm_RMAP

import pandas as pd
import numpy as np

nmr = nm_RMAP.RMAPChannelSelector()

# load rmap:
PATH_RMAP = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\rmap_func.nii"
PATH_CONN = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\real_time_test\get_rmap_ch_selection\OUT_func_conn\GSP 1000 (Yeo 2011)_Full Set (Yeo 2011)"
rmap = nmr.load_fingerprint(PATH_RMAP)
fps = nmr.load_all_fingerprints(PATH_CONN)

corr_out = []
for idx, fp in enumerate(fps[0]):
    corr_out.append(nmr.get_corr_numba(rmap.flatten(), fps[1][idx].flatten()))


PATH_CONN = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Berlin\functional connectivity"



df_per = pd.read_csv(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\real_time_test\model_output\per_ecog_MedOn.csv")

# iterate through rmap and select the channels with sub and ch

l_fp_use = []
l_per_use = []

for idx, row in df_per.iterrows():
    for idx_fp, fp in enumerate(fps[0]):
        # the nomenclature unfortunately changed from the time the connectivity maps were computed
        # e.g. sometimes left and right was changed...
        if row["sub"][4:] in fp and row["ch"][row["ch"].find("SMC_AT")-3:row["ch"].find("SMC_AT")] in fp:
            l_fp_use.append(np.nan_to_num(fps[1][idx_fp]))
            l_per_use.append(row["per"])

#rmap = nmr.get_RMAP(l_per_use, np.vstack(l_fp_use))
rmap = nmr.calculate_RMap_numba(l_fp_use, l_per_use)

# when the rmap was calculated; select the one with highest correlattion prediction performances
nmr.save_Nii(np.nan_to_num(rmap), nmr.affine, "rmap_save_med_on.nii")