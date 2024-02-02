import pandas as pd
import numpy as np
from py_neuromodulation import nm_stats
import mne
from matplotlib import pyplot as plt
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d

# get examplar time series for sw motivation
PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin\sub-002\ses-EcogLfpMedOff01\ieeg\sub-002_ses-EcogLfpMedOff01_task-SelfpacedRotationR_acq-StimOff_run-01_ieeg.vhdr"
raw = mne.io.read_raw_brainvision(PATH_RUN, preload=True)


data = raw.get_data()

raw_snippet = -data[0, 375794:379826]
times = raw.times[375794:379826] - raw.times[375794]

plt.figure(figsize=(8, 3), dpi=300)
plt.subplot(1, 3, 1)
plt.plot(times, raw_snippet)
plt.title("Raw rotameter signal")
plt.ylabel("Amplitude [a.u.]")
plt.xlabel("Time [s]")

plt.subplot(1, 3, 2)
binary_signal = raw_snippet > 0.05
plt.plot(times, binary_signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [a.u.]")
plt.title("Binarized signal")

plt.subplot(1, 3, 3)
filtered_signal = gaussian_filter1d(np.array(binary_signal, dtype=float), sigma=0.5*137)
plt.plot(times, filtered_signal)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [a.u.]")
plt.title("Gaussian filtered signal")
plt.tight_layout()
plt.savefig("figure\\example_gaussian_filter_motivation.pdf", bbox_inches="tight")
plt.show()







plt.subplot(1, 3, 2)
plt.plot(-data > )

