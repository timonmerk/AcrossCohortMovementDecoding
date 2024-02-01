import mne_bids
import mne
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_IO, nm_define_nmchannels, nm_stream_offline
import py_neuromodulation as nm
from multiprocessing import Pool
from scipy import stats
import sys
import os

PATH_IN_BASE = r"C:\Users\ICN_admin\Documents\Datasets"


def est_features_run(PATH_RUN):
    PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_all\Washington"
    RUN_NAME = os.path.basename(PATH_RUN[:-4])  # cut .mat
    if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True:
        print("path exists")
        return
    dat = nm_IO.loadmat(PATH_RUN)
    label = dat["stim"]

    sfreq = 1000
    ch_names = [f"ECOG_{i}" for i in range(dat["data"].shape[1])]
    ch_names = ch_names + ["mov"]
    ch_types = ["ecog" for _ in range(dat["data"].shape[1])]
    ch_types = ch_types + ["misc"]
    data_uv = dat["data"] * 0.0298  # see doc description

    data = np.concatenate((data_uv, np.expand_dims(label, axis=1)), axis=1).T
    return data.shape[1] / (1000 * 60), np.where(data[-1, :] == 12)[0].shape[0] / (
        1000 * 60
    )


if __name__ == "__main__":
    PATH_DATA = os.path.join(PATH_IN_BASE, r"Washington\motor_basic\data")
    run_files_Washington = [
        os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if "_mot_t_h" in f
    ]

    overall_duration = []
    movemetn_duration = []
    for PATH_RUN in run_files_Washington:
        res = est_features_run(PATH_RUN)
        overall_duration.append(res[0])
        movemetn_duration.append(res[1])
