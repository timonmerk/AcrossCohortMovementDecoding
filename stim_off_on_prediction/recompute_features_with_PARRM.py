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

    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_parrm_removed"
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
    parrm = nm_artifacts.PARRMArtifactRejection(data[ecog_idx, :], sfreq, 130)
    filtered_data = parrm.filter_data()

    PLT_ = True
    if PLT_ is True:

        # plot here a simple welch PSD of the filtered data using scipy

        filtered_data_bandstop = mne.filter.filter_data(
            data=data[ecog_idx, :],
            sfreq=sfreq,
            l_freq=160,
            h_freq=100,
            method="fir",
            verbose=False,
            l_trans_bandwidth=5,
            h_trans_bandwidth=5,
        )

        from scipy.signal import welch

        ch_idx = 4

        # None
        idx_mov = np.where(data[-2, :] == 1)[0]
        idx_rest = np.where(data[-2, :] == 0)[0]

        plt.figure(figsize=(12, 4), dpi=300)
        plt.suptitle("PSD Movement and Rest")

        plt.subplot(1, 3, 1)
        f, Pxx_mov = welch(data[ecog_idx[ch_idx], idx_mov], fs=sfreq, nperseg=1024)
        f, Pxx_rest = welch(data[ecog_idx[ch_idx], idx_rest], fs=sfreq, nperseg=1024)
        plt.plot(f, 10 * np.log10(Pxx_rest), color="b", label="Rest")
        plt.plot(f, 10 * np.log10(Pxx_mov), color="r", label="Move")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.title("Without filter")
        plt.xscale("log")
        plt.legend()
        plt.show()

        plt.subplot(1, 3, 2)
        f, Pxx_mov = welch(filtered_data[ch_idx, idx_mov], fs=sfreq, nperseg=1024)
        f, Pxx_rest = welch(filtered_data[ch_idx, idx_rest], fs=sfreq, nperseg=1024)
        plt.plot(f, 10 * np.log10(Pxx_rest), color="b", label="Rest")
        plt.plot(f, 10 * np.log10(Pxx_mov), color="r", label="Move")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.title("PARRM Filtered")
        # use log x axis
        plt.xscale("log")
        plt.legend()

        plt.subplot(1, 3, 3)
        f, Pxx_mov = welch(
            filtered_data_bandstop[ch_idx, idx_mov], fs=sfreq, nperseg=1024
        )
        f, Pxx_rest = welch(
            filtered_data_bandstop[ch_idx, idx_rest], fs=sfreq, nperseg=1024
        )
        plt.plot(f, 10 * np.log10(Pxx_rest), color="b", label="Rest")
        plt.plot(f, 10 * np.log10(Pxx_mov), color="r", label="Move")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.title("Bandstop Filter")
        plt.xscale("log")
        plt.legend()

        plt.tight_layout()
        plt.savefig("figure\\PSD_STIM_ON_MOVE_PARRM_BANDSTOP_comparison.pdf")
        plt.show()

        # idea: plot separate plots for PARRM, bandstop, None
        plt.figure(figsize=(12, 4), dpi=300)

        # None
        idx_mov = np.where(data[-2, :] == 1)[0]
        idx_rest = np.where(data[-2, :] == 0)[0]

        f, Pxx_mov = welch(data[ecog_idx[ch_idx], idx_mov], fs=sfreq, nperseg=1024)
        f, Pxx_rest = welch(data[ecog_idx[ch_idx], idx_rest], fs=sfreq, nperseg=1024)

        # plt.subplot(1, 3, 1)
        plt.plot(f, 10 * np.log10(Pxx_mov) - 10 * np.log10(Pxx_rest), label="Raw Data")
        # plt.title("Raw data");
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.xlim([0, 700])
        plt.ylim([-6, 8])

        f, Pxx_mov = welch(
            filtered_data_bandstop[ch_idx, idx_mov], fs=sfreq, nperseg=1024
        )
        f, Pxx_rest = welch(
            filtered_data_bandstop[ch_idx, idx_rest], fs=sfreq, nperseg=1024
        )

        # plt.subplot(1, 3, 2)
        # plt.plot(f, 10 * np.log10(Pxx_mov) - 10 * np.log10(Pxx_rest), label="Bandstop filtered data")
        # plt.title("Bandstop filtered data");
        # plt.xlabel("Frequency [Hz]"); plt.ylabel("Power [dB]"); plt.xlim([0, 700]); plt.ylim([-6, 8])

        f, Pxx_mov = welch(filtered_data[ch_idx, idx_mov], fs=sfreq, nperseg=1024)
        f, Pxx_rest = welch(filtered_data[ch_idx, idx_rest], fs=sfreq, nperseg=1024)

        # plt.subplot(1, 3, 3)
        plt.plot(
            f,
            10 * np.log10(Pxx_mov) - 10 * np.log10(Pxx_rest),
            label="PARRM filtered data",
        )
        # plt.title("PARRM filtered data");
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.xlim([0, 700])
        plt.ylim([-6, 8])
        # plot a dotted horizontal line at 0
        plt.axhline(y=0, color="k", linestyle="--", label="Baseline")
        plt.legend()

        plt.suptitle("Movement minus rest Welch PSD")
        plt.tight_layout()

        plt.figure(figsize=(12, 4), dpi=300)

        for idx, spec_ in enumerate(["mov", "rest", "all"]):
            plt.subplot(1, 3, idx + 1)
            if spec_ == "mov":
                idx_ = np.where(data[-2, :] == 1)[0]
            elif spec_ == "rest":
                idx_ = np.where(data[-2, :] == 0)[0]
            else:
                idx_ = np.arange(data.shape[1])

            f, Pxx_parrm = welch(filtered_data[ch_idx, idx_], fs=sfreq, nperseg=1024)
            f, Pxx = welch(data[ecog_idx[ch_idx], idx_], fs=sfreq, nperseg=1024)
            f, Pxx_bandstop_filtered = welch(
                filtered_data_bandstop[ch_idx, idx_], fs=sfreq, nperseg=1024
            )

            plt.plot(f, 10 * np.log10(Pxx), label="Raw")
            plt.plot(f, 10 * np.log10(Pxx_parrm), label="PARRM filtered")
            plt.plot(f, 10 * np.log10(Pxx_bandstop_filtered), label="Bandstop filtered")

            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dB]")
            plt.title(f"PSD {spec_} data")
            plt.xlim([0, 200])
            plt.legend()

        plt.suptitle("Welch PSD of PARRM filtered data")
        plt.tight_layout()

        # plot here also now a spectogram of the filtered data

        from scipy.signal import spectrogram

        spec = spectrogram(
            filtered_data[0, :], fs=sfreq, nperseg=1024, noverlap=1024 - 100
        )
        plt.figure(figsize=(5, 3), dpi=300)
        plt.pcolormesh(
            spec[1], spec[0], 10 * np.log10(spec[2]), shading="gouraud", cmap="inferno"
        )
        plt.colorbar(label="dB")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.title("Spectrogram of PARRM filtered data")

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

    # clean the data here:

    # cut for Berlin sub012 the last ECoG channel, due to None coordinates
    # if "Berlin" in PATH_RUN and "sub-012" in PATH_RUN:
    #    coord_list = coord_list[:-3]
    #    coord_names = coord_names[:-3]
    #    data = data[:-3, :]

    data[ecog_idx, :] = filtered_data

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

        est_features_run(run_files_Berlin[0])

        # setup parallel processing using joblib
        from joblib import Parallel, delayed

        Parallel(n_jobs=len(run_files_Berlin[1:]))(
            delayed(est_features_run)(run) for run in run_files_Berlin[1:]
        )

    cohort_runner = nm_cohortwrapper.CohortRunner(
        outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_stim_parrm_removed",
        cohorts={"Berlin": ""},
    )

    cohort_runner.cohort_wrapper_read_all_grid_points(read_channels=True)
