import mne_bids
import mne
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_IO, nm_define_nmchannels, nm_stream_offline
from py_neuromodulation import nm_artifacts
from py_neuromodulation import nm_cohortwrapper
import py_neuromodulation as nm
from scipy import stats
import os
import pandas as pd
from bids import BIDSLayout


def est_features_run(PATH_RUN):
    def set_settings(settings: dict):
        settings["features"]["fft"] = True
        settings["features"]["fooof"] = False
        settings["features"]["return_raw"] = False
        settings["features"]["raw_hjorth"] = False
        settings["features"]["sharpwave_analysis"] = False
        settings["features"]["nolds"] = False
        settings["features"]["bursts"] = False
        settings["features"]["coherence"] = False

        settings["preprocessing"] = ["raw_resampling", "notch_filter", "re_referencing"]

        settings["postprocessing"]["feature_normalization"] = True
        settings["postprocessing"]["project_cortex"] = False
        settings["postprocessing"]["project_subcortex"] = False

        return settings

    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_bandstop_filtered"
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
    PATH_OUT = os.path.join(PATH_OUT_BASE, "Berlin")

    RUN_NAME = os.path.basename(PATH_RUN)[:-5]
    if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True:
        print("path exists")
    #    return
    (
        raw,
        data,
        sfreq,
        line_noise,
        coord_list,
        coord_names,
    ) = nm_IO.read_BIDS_data(PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg")

    ecog_idx = [
        idx
        for idx, ch in enumerate(raw.ch_names)
        if "ECOG" in ch and ch not in raw.info["bads"]
    ]
    # parrm = nm_artifacts.PARRMArtifactRejection(data[ecog_idx, :], sfreq, 130)
    # filtered_data = parrm.filter_data()
    # bandstop filter the data between 100 Hz and 160 Hz

    filtered_data = mne.filter.filter_data(
        data=data[ecog_idx, :],
        sfreq=sfreq,
        l_freq=160,
        h_freq=100,
        method="fir",
        verbose=False,
        l_trans_bandwidth=5,
        h_trans_bandwidth=5,
    )

    PLT_ = True
    if PLT_ is True:
        plt.figure(figsize=(5, 3), dpi=300)
        start = 123000 - 300
        end = 129000 - 300
        plt.title("PARRM comparison")

        plt.plot(
            np.arange(start, end) / sfreq,
            data[ecog_idx[0], start:end],
            label="raw",
            linewidth=0.5,
        )
        plt.plot(
            np.arange(start, end) / sfreq,
            filtered_data[0, start:end],
            label="PARRM filtered",
            linewidth=0.5,
        )
        plt.plot(
            np.arange(start, end) / sfreq,
            data[-1, start:end] * 50,
            label="Movement",
            linewidth=0.5,
        )
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [a.u.]")

        # plot also the stft
        from scipy import signal, stats

        f, t, Zxx = signal.stft(data[ecog_idx[0], start:end], sfreq, nperseg=sfreq)
        plt.figure(figsize=(5, 3), dpi=300)
        plt.pcolormesh(t, f, stats.zscore(np.abs(Zxx), axis=0), shading="gouraud")
        plt.title("STFT Magnitude")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar()
        plt.ylim(0, 200)
        plt.clim(-3, 10)
        plt.show()

        # and now from the filtered_data
        f, t, Zxx = signal.stft(
            filtered_data[ecog_idx[0], start:end], sfreq, nperseg=sfreq
        )
        plt.figure(figsize=(5, 3), dpi=300)
        plt.pcolormesh(t, f, stats.zscore(np.abs(Zxx), axis=0), shading="gouraud")
        plt.title("STFT Magnitude")
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        plt.colorbar()
        plt.ylim(0, 200)
        plt.clim(-3, 10)
        plt.show()

    data[ecog_idx, :] = filtered_data

    # clean the data here:

    # cut for Berlin sub012 the last ECoG channel, due to None coordinates
    # if "Berlin" in PATH_RUN and "sub-012" in PATH_RUN:
    #    coord_list = coord_list[:-3]
    #    coord_names = coord_names[:-3]
    #    data = data[:-3, :]

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog",),
        target_keywords=("SQUARED_EMG", "mov", "squared", "label"),
    )

    settings = nm.nm_settings.get_default_settings()
    settings = nm.nm_settings.reset_settings(settings)

    settings = set_settings(settings)

    try:
        stream = nm.Stream(
            settings=settings,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=True,
            sfreq=sfreq,
            line_noise=line_noise,
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )
    except:
        print(f"could not run {RUN_NAME}")


if __name__ == "__main__":

    COMPUTE_FEATURES = True

    if COMPUTE_FEATURES is True:
        # first check which runs need to be computed
        df = pd.read_csv("stim_off_on_prediction\\df_STIM_ON_OFF_predict.csv")
        subjects = df["Subject"]

        PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
        layout = BIDSLayout(PATH_BIDS)

        run_files_Berlin = layout.get(
            task=["SelfpacedRotationR", "SelfpacedRotationL", "SelfpacedForceWheel"],
            extension=".vhdr",
        )

        run_files_Berlin = [
            f.path
            for f in run_files_Berlin
            if "StimOn" in f.path and "EL016" not in f.path and "EL017" not in f.path
        ]

        # est_features_run(run_files_Berlin[0])

        # setup parallel processing using joblib
        from joblib import Parallel, delayed

        Parallel(n_jobs=len(run_files_Berlin[:]))(
            delayed(est_features_run)(run) for run in run_files_Berlin[:]
        )

    cohort_runner = nm_cohortwrapper.CohortRunner(
        outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_bandstop_filtered",
        cohorts={"Berlin": ""},
    )

    cohort_runner.cohort_wrapper_read_all_grid_points(read_channels=True)
