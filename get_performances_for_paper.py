import pandas as pd
import numpy as np
from py_neuromodulation import nm_stats
import mne
from matplotlib import pyplot as plt
import pickle
import numpy as np

# get recording durattion per patient
ch_all = np.load("features_out_fft\\channel_all.npy", allow_pickle=True).item()
df_durations = pd.DataFrame(
    columns=["cohort", "sub", "duration", "duration_mov", "duration_rest"]
)

for cohort in ch_all.keys():
    for sub in ch_all[cohort].keys():
        ch = list(ch_all[cohort][sub].keys())[0]
        comb_data = []
        comb_label = []
        for run in ch_all[cohort][sub][ch].keys():
            comb_data.append(ch_all[cohort][sub][ch][run]["data"])
            comb_label.append(ch_all[cohort][sub][ch][run]["label"])
        comb_data = np.concatenate(comb_data, axis=0)
        comb_label = np.concatenate(comb_label, axis=0)

        total_duration = comb_data.shape[0] / (10 * 60)  # min
        duration_movement = comb_data[np.where(comb_label)[0], :].shape[0] / (10 * 60)
        duration_rest = comb_data[np.where(np.logical_not(comb_label))[0], :].shape[
            0
        ] / (10 * 60)

        df_dict = pd.DataFrame(
            {
                "cohort": cohort,
                "sub": sub,
                "duration": total_duration,
                "duration_mov": duration_movement,
                "duration_rest": duration_rest,
            },
            index=[0],
        )
        df_durations = pd.concat([df_durations, df_dict], ignore_index=True)

df_durations.to_csv(
    "read_performances\\df_time_all_mov_cohorts_durations.csv", index=False
)

# get run numbers:
with open("read_performances\\df_all.p", "rb") as handle:
    df = pickle.load(handle)  # theta, alpha, low beta, high beta, low gamma, broadband

# number of total recordings
print(df.query("cohort == 'Berlin'").groupby(["cohort", "sub", "run"]).mean().shape[0])
runs_Berlin = list(
    df.query("cohort == 'Berlin'")
    .groupby(["cohort", "sub", "run"])
    .mean()
    .reset_index()["run"]
)

# print number of Berlin MedOn recordings
print(len([r for r in runs_Berlin if "MedOn" in r]))

runs = list(df.groupby(["cohort", "sub", "run"]).mean().reset_index()["run"])

# number of Pittsburgh recording
print(
    df.query("cohort == 'Pittsburgh'").groupby(["cohort", "sub", "run"]).mean().shape[0]
)

print(
    df.query("cohort == 'Washington'").groupby(["cohort", "sub", "run"]).mean().shape[0]
)

# get example time series for STIM ON and OFF
PATH_STIM_ON = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff03\ieeg\sub-002_ses-EcogLfpMedOff03_task-SelfpacedRotationR_acq-StimOn_run-01_ieeg.vhdr"
PATH_STIM_OFF = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff03\ieeg\sub-002_ses-EcogLfpMedOff03_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"

OFFSET = int(1375 * 67.5 - 5)

raw = mne.io.read_raw_brainvision(PATH_STIM_ON, preload=True)
ch_names = [c for c in raw.ch_names if "ECOG" in c or "SQUARED_ROTATION" in c]
raw_p = raw.pick(picks=ch_names)
raw_p = raw_p.filter(l_freq=2, h_freq=None)

dat_plt_on = raw_p.get_data()[5, OFFSET : OFFSET + int(raw_p.info["sfreq"]) * 10]

OFFSET = int(1375 * 202.5 - 5)

raw = mne.io.read_raw_brainvision(PATH_STIM_OFF, preload=True)
ch_names = [c for c in raw.ch_names if "ECOG" in c or "SQUARED_ROTATION" in c]
raw_p = raw.pick(picks=ch_names)
raw_p = raw_p.filter(l_freq=2, h_freq=None)
dat_plt_off = raw_p.get_data()[5, OFFSET : OFFSET + int(raw_p.info["sfreq"]) * 10]


plt.figure(figsize=(4, 3), dpi=300)
plt.plot(dat_plt_off, label="STIM OFF", linewidth=1)
plt.plot(dat_plt_on + 300, label="STIM ON", linewidth=1)
plt.xticks(np.arange(0, 10 * 1375, 1375), np.arange(0, 10 * 1375, 1375) / 1375)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude a.u.")
plt.title("Time series example")
plt.legend()

plt.savefig(
    "figure\\example_stim_on_off_time_trace.pdf",
    bbox_inches="tight",
)

raw_p.plot()


# get statistics


# mean performances individual channels
df = pd.read_csv("read_performances\\df_ch_performances.csv")
print(f"{np.mean(df['performance_test'])} {np.std(df['performance_test'])}")

df_grouped = df.groupby(["cohort", "sub"]).max()
print(
    f"{np.mean(df_grouped['performance_test'])} {np.std(df_grouped['performance_test'])}"
)
print(
    f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}"
)

# mean performances grid points
df = pd.read_csv("read_performances\\df_grid_point_performances.csv")
print(f"{np.mean(df['performance_test'])} {np.std(df['performance_test'])}")

df_grouped = df.groupby(["cohort", "sub"]).max()
print(
    f"{np.mean(df_grouped['performance_test'])} {np.std(df_grouped['performance_test'])}"
)
print(
    f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}"
)

# best r-map selected channels
df = pd.read_csv("read_performances\\df_best_func_rmap_ch.csv")
df_grouped = df.groupby(["cohort", "sub"]).max()
print(
    f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}"
)

# group wise best detection rate individual channels
df = pd.read_csv("read_performances\\df_ch_performances.csv")
df_grouped = df.groupby(["cohort", "sub"]).max()
print(
    f"{np.mean(df_grouped['mov_detection_rates_test'])} {np.std(df_grouped['mov_detection_rates_test'])}"
)

# permutation test STIM conditions
df = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF.csv")

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
nm_stats.permutationTest_relative(x_, y_, False, None, 5000)
# p=0.25

x_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF->ON Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
# p=0.012

x_ = np.array(df[df["Model Type"] == "STIM OFF"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM ON->OFF Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
# p=0.03

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM OFF->ON Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
# p=0.09

x_ = np.array(df[df["Model Type"] == "STIM ON"]["Test Performance"])
y_ = np.array(df[df["Model Type"] == "STIM ON->OFF Predict"]["Test Performance"])
print(nm_stats.permutationTest_relative(x_, y_, False, None, 5000))
# p=0.4

# read movement detection results for stim
df = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_with_detection_rates.csv")

print(
    f"{df[df['Model Type'] == 'STIM OFF->ON Predict']['Test Performance Detection Rate'].mean()} {df[df['Model Type'] == 'STIM OFF->ON Predict']['Test Performance Detection Rate'].std()}"
)
