from matplotlib import pyplot as plt

from sklearn import linear_model
import seaborn as sb

#import umap

import numpy as np
import os
import pandas as pd

ch_all = np.load(
    os.path.join(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft", "channel_all.npy"),
    allow_pickle="TRUE",
).item()

df_best_rmap = pd.read_csv(r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\across patient running\RMAP\df_best_func_rmap_ch.csv")

df_performances = pd.read_csv("plt_on_cortex\\df_ch_performances.csv")

# plotting correlation matrices makes only sense when averaging across patients

d_out = {}

for cohort in ch_all.keys():
    for sub in ch_all[cohort].keys():
        for ch in ch_all[cohort][sub].keys():
            runs = list(ch_all[cohort][sub][ch].keys())
            runs = [r for r in runs if "StimOn" not in r]
            if len(runs) > 1:
                dat_concat = np.concatenate([ch_all[cohort][sub][ch][run]["data"] for run in runs], axis=0)
                lab_concat = np.concatenate([ch_all[cohort][sub][ch][run]["label"] for run in runs], axis=0)
            else:
                dat_concat = ch_all[cohort][sub][ch][runs[0]]["data"]
                lab_concat = ch_all[cohort][sub][ch][runs[0]]["label"]


            corr_ = np.corrcoef(dat_concat.T)
            model = linear_model.LogisticRegression(class_weight="balanced").fit(dat_concat, lab_concat)
            model.coef_

            d_out[f"{cohort}_{sub}"] = {}
            d_out[f"{cohort}_{sub}"]["corr"] = corr_
            d_out[f"{cohort}_{sub}"]["lm_coef"] = model.coef_

            d_out[f"{cohort}_{sub}"]["mean_features_mov"] = dat_concat[np.where(lab_concat)[0], :].mean(axis=0)
            d_out[f"{cohort}_{sub}"]["mean_features_rest"] = dat_concat[np.where(np.logical_not(lab_concat))[0], :].mean(axis=0)

# the individual fbands need to be saved to a csv file and then plotted in matlab

fband_names = ["theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA"]

theta_rest = []
alpha_rest = []
low_beta_rest = []
high_beta_rest = []
low_gamma_rest = []
high_gamma_rest = []
HFA_rest = []

theta_mov = []
alpha_mov = []
low_beta_mov = []
high_beta_mov = []
low_gamma_mov = []
high_gamma_mov = []
HFA_mov = []


for idx, row in df_performances.iterrows():
    for fb_idx, f_band in enumerate(fband_names):
        d_out[row["cohort"]][row["sub"]][row["ch"]]["mean_features_mov"][fb_idx]
        


beta_diff = np.zeros([len(d_out.keys()), len(d_out.keys())])

i, j = 0, 0

for sub_name_1, d_1 in d_out.items():
    for sub_name_2, d_2 in d_out.items():
        beta_diff[i, j] = d_out[sub_name_1]["mean_features_mov"][3] - d_out[sub_name_2]["mean_features_mov"][3]
        j +=1
    j=0
    i+=1

# original idea: plot feature differences

# does the number of samples matter for logistic regression?

# simple difference between mean beta mov/rest


beta_diff = np.zeros([len(d_out.keys()), len(d_out.keys())])

i, j = 0, 0

for sub_name_1, d_1 in d_out.items():
    for sub_name_2, d_2 in d_out.items():
        beta_diff[i, j] = d_out[sub_name_1]["mean_features_mov"][3] - d_out[sub_name_2]["mean_features_mov"][3]
        j +=1
    j=0
    i+=1

plt.figure(figsize=(5,5))
plt.imshow(beta_diff, aspect="auto")
plt.title("High beta norm")
plt.show()

# calc distance between vectors:
from scipy.spatial.distance import cosine

out_norm = np.zeros([len(d_out.keys()), len(d_out.keys())])
out_cos = np.zeros([len(d_out.keys()), len(d_out.keys())])
i = 0
j = 0
for sub_name_1, d_1 in d_out.items():
    for sub_name_2, d_2 in d_out.items():
        out_norm[i, j] = np.linalg.norm(d_out[sub_name_1]["lm_coef"] - d_out[sub_name_2]["lm_coef"])
        out_cos[i, j] = cosine(d_out[sub_name_1]["lm_coef"], d_out[sub_name_2]["lm_coef"])
        j +=1
    j=0
    i+=1

plt.figure(figsize=(5,5))
plt.imshow(out_norm, aspect="auto")
plt.title("Eucledian Norm")
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(out_cos, aspect="auto")
plt.title("Cosine Distance")
plt.show()


fband_names = ["theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "HFA"]

# 1. plot mean correlation matrix

plt.figure(figsize=(5,5), dpi=300)
arr = np.array([d_out[sub_name]["corr"] for sub_name, d_ in d_out.items()]).mean(axis=0)
plt.imshow(arr, aspect="auto")
plt.xticks(np.arange(7), fband_names, rotation=90)
plt.yticks(np.arange(7), fband_names, )
plt.title("Mean correlation matrix")
cbar = plt.colorbar()
cbar.set_label("Pearson Corr. Coeff.")
plt.tight_layout()
plt.savefig("mean_corr_matrix_features.pdf")


# calculate distances between matrices

# 2. boxplot features, draw lines between them

df_plt = []

for sub_name, d_ in d_out.items():
    for f_band_idx, f_band in enumerate(fband_names):
        df_plt.append(pd.Series({
            "f_band": f_band,
            "feature_val" : d_out[sub_name]["mean_features_mov"][f_band_idx],
            "cond" : "mov",
            "sub" : sub_name
        }))
        df_plt.append(pd.Series({
            "f_band": f_band,
            "feature_val" : d_out[sub_name]["mean_features_rest"][f_band_idx],
            "cond" : "rest",
            "sub" : sub_name
        }))
df_plt = pd.DataFrame(df_plt)

df_plt.to_csv("mean_feature_df_plt.csv")

from plotnine import *

plt.figure(figsize=(5,5), dpi=300)
sb.boxplot(x="f_band", y="feature_val", hue="cond", data=df_plt)
sb.lineplot(
     x="f_band", y="feature_val", units="cond",
    color=".7", estimator=None, data=df_plt
)

