import os
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats

df_rmap_corr = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

with open(os.path.join("read_performances", "df_all.p"), "rb") as handle:
    df_all = pickle.load(handle)

df = df_all.query("ch_type == 'electrode ch' and cohort != 'Washington'")

# some patients have stim on and off runs
df_STIMON = df.query("run.str.contains('StimOn')")
df_STIMOFF = df.query("run.str.contains('StimOff') and cohort == 'Berlin'")

# for each subject select best RMAP channel
df_comp_STIM = pd.DataFrame()
for sub in df_STIMON["sub"].unique():
    ch = df_rmap_corr.query("cohort=='Berlin' and sub == @sub").iloc[0].ch
    per_ON = df_STIMON.query("sub == @sub and ch == @ch")["performance_test"].mean()
    per_OFF = df_STIMOFF.query("sub == @sub and ch == @ch")["performance_test"].mean()
    df_comp_STIM = df_comp_STIM.append(
        {
            "performance": per_ON,
            "sub": sub,
            "STIM_Condition": "STIM ON",
        },
        ignore_index=True,
    )

    df_comp_STIM = df_comp_STIM.append(
        {
            "performance": per_OFF,
            "sub": sub,
            "STIM_Condition": "STIM OFF",
        },
        ignore_index=True,
    )

df_comp_STIM.to_csv("df_comp_STIM.csv")

gt, p = nm_stats.permutationTest(
    list(df_comp_STIM.query("STIM_Condition == 'STIM ON'")["performance"]),
    list(df_comp_STIM.query("STIM_Condition == 'STIM OFF'")["performance"]),
    False,
    "R^2",
    5000,
)

plt.figure(figsize=(6, 5), dpi=300)
sb.barplot(
    data=df_comp_STIM, x="sub", y="performance", hue="STIM_Condition", palette="viridis"
)
plt.title(f"STIM ON / OFF Comparison Berlin subjects\np={np.round(p,3)}")
plt.savefig(
    os.path.join("figure", "stim_off_on_comp.pdf"),
    bbox_inches="tight",
)
