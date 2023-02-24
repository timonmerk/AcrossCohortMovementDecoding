import pandas as pd
import numpy as np
from py_neuromodulation import nm_stats
import mne
from matplotlib import pyplot as plt

# get example time series
PATH_STIM_ON = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff03\ieeg\sub-002_ses-EcogLfpMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg.vhdr"
PATH_STIM_OFF = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff03\ieeg\sub-002_ses-EcogLfpMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"

raw = mne.io.read_raw_brainvision(PATH_STIM_ON)
ch_names = [c for c in raw.ch_names if "ECOG" in c]
raw_p = raw.pick(picks=ch_names)

dat_plt_on = raw_p.get_data()[1, 5000:5000+int(raw_p.info["sfreq"])]

raw = mne.io.read_raw_brainvision(PATH_STIM_OFF)
ch_names = [c for c in raw.ch_names if "ECOG" in c]
raw_p = raw.pick(picks=ch_names)
dat_plt_off = raw_p.get_data()[1, 5000:5000+int(raw_p.info["sfreq"])]


plt.figure(figsize=(4, 3), dpi=300)
plt.plot(dat_plt_off, label="OFF")
plt.plot(dat_plt_on, label="ON")

plt.savefig(
    "figure\\example_stim_on_off_time_trace.pdf",
    bbox_inches="tight",
)

raw_p.plot()

# mean performances individual channels
df = pd.read_csv("plt_on_cortex\\df_ch_performances.csv")
print(f"{np.mean(df['performance_test'])} {np.std(df['performance_test'])}")

df_grouped = df.groupby(["cohort", "sub"]).max()
print(f"{np.mean(df_grouped['performance_test'])} {np.std(df_grouped['performance_test'])}")
print(f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}")

# mean performances grid points
df = pd.read_csv("read_performances\\df_grid_point_performances.csv")
print(f"{np.mean(df['performance_test'])} {np.std(df['performance_test'])}")

df_grouped = df.groupby(["cohort", "sub"]).max()
print(f"{np.mean(df_grouped['performance_test'])} {np.std(df_grouped['performance_test'])}")
print(f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}")

# best r-map selected channels
df = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")
df_grouped = df.groupby(["cohort", "sub"]).max()
print(f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}")

# group wise best detection rate individual channels
df = pd.read_csv("read_performances\\df_ch_performances.csv")
df_grouped = df.groupby(["cohort", "sub"]).max()
print(f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}")

# permutation test STIM conditions
df = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF.csv")

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
nm_stats.permutationTest_relative(x_, y_, False, None, 5000)
#p=0.25

x_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF->ON Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
#p=0.012

x_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM ON->OFF Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
#p=0.03

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF->ON Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
#p=0.09

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM ON->OFF Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
#p=0.4

# read here cross-validation performances
