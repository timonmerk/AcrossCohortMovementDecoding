from matplotlib import pyplot as plt
import os
import numpy as np
import pandas as pd

model_name = "LM"

PATH_CROSS_VAL_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"


p = np.load(
    os.path.join(PATH_CROSS_VAL_BASE, f"{model_name}_performance_p2p_RMAP.npy"),  # LM
    allow_pickle="TRUE",
).item()


sub_list = []
for cohort in list(p.keys()):
    for sub in list(p[cohort].keys()):
        sub_list.append(f"{cohort}_{sub}")


per_per = np.load(os.path.join("figure", "p2p_arr_LM.npy"), allow_pickle="TRUE")
# read UPDRS results

df_updrs = pd.read_csv("df_updrs.csv")
df_updrs["sub_cohort_name"] = df_updrs["cohort"] + "_" + df_updrs["sub"]

idx_ = []
for sub_ in sub_list:
    idx_.append(df_updrs[df_updrs["sub_cohort_name"] == sub_].index[0])

df_updrs_sorted = df_updrs.iloc[idx_].reset_index()
idx_updrs_sort = np.array(df_updrs_sorted["UPDRS_total"]).argsort()
per_per_sorted = per_per[idx_updrs_sort, :][:, idx_updrs_sort]
per_per_sorted = per_per_sorted[
    :-2, :-2
]  # two subjects (Pittsburgh 003 and Berlin 001 don't have UPDRS)

plt.figure(figsize=(6, 6), dpi=300)
plt.imshow(per_per_sorted, aspect="auto")
plt.xticks(
    np.arange(len(sub_list))[:-2], np.array(sub_list)[idx_updrs_sort][:-2], rotation=90
)
plt.yticks(np.arange(len(sub_list))[:-2], np.array(sub_list)[idx_updrs_sort][:-2])
cbar = plt.colorbar()
cbar.set_label("Balanced Accuracy")
plt.title(f"{model_name} patient to patient predictions UPDRS sorted")
plt.xlabel("Train Patient")
plt.ylabel("Test Patient")
plt.clim(
    0.5,
)
plt.savefig(
    f"p2p_{model_name}_UPDRS_sortedd.pdf",
    bbox_inches="tight",
)

# no sorted

plt.figure(figsize=(6, 6), dpi=300)
plt.imshow(per_per, aspect="auto")
plt.xticks(np.arange(len(sub_list)), sub_list, rotation=90)
plt.yticks(np.arange(len(sub_list)), sub_list)
cbar = plt.colorbar()
cbar.set_label("Balanced Accuracy")
plt.title(f"{model_name} patient to patient predictions")
plt.xlabel("Train Patient")
plt.ylabel("Test Patient")
plt.clim(
    0.5,
)
plt.savefig(
    f"p2p_{model_name}.pdf",
    bbox_inches="tight",
)
