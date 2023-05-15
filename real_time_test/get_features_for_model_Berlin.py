import mne_bids
import mne
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_IO, nm_define_nmchannels, nm_stream_offline
import py_neuromodulation as nm
from multiprocessing import Pool
from scipy import stats
import os

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

        settings["preprocessing"] = [
            "raw_resampling",
            "notch_filter",
            "re_referencing"
        ]

        settings["postprocessing"]["feature_normalization"] = True
        settings["postprocessing"]["project_cortex"] = False
        settings["postprocessing"]["project_subcortex"] = False

        return settings

    PATH_OUT_BASE = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_no_kf"

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
    PATH_OUT = os.path.join(PATH_OUT_BASE, "Berlin")

    RUN_NAME = os.path.basename(PATH_RUN)[:-5]
    if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True:
        print("path exists")
        return
    (raw, data, sfreq, line_noise, coord_list, coord_names,) = nm_IO.read_BIDS_data(
        PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg"
    )

    nm_channels = nm_define_nmchannels.set_channels(
        ch_names=raw.ch_names,
        ch_types=raw.get_channel_types(),
        reference="default",
        bads=raw.info["bads"],
        new_names="default",
        used_types=("ecog", "seeg", "dbs"),
        target_keywords=(
            "SQUARED_EMG",
            "mov",
            "squared",
            "label",
            "ANALOG",
            "ANALOG_R_ROTA_CH",
            "SQUARED_ROTATION"
        ),
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
            line_noise=line_noise
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )
    except:
        print(f"could not run {RUN_NAME}")

if __name__ == "__main__":

    # collect all run files
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
    layout = BIDSLayout(PATH_BIDS)

    run_files_Berlin = layout.get(
        task=["SelfpacedRotationR", "SelfpacedRotationL"],
        extension=".vhdr",
    )
    run_files_Berlin = [f.path for f in run_files_Berlin]
    #est_features_run(run_files_Berlin[10])

    #for run_file in run_all:
    #    est_features_run(run_file)

    #est_features_run(run_files_Berlin[0])
    # run parallel Pool
    pool = Pool(processes=len(run_files_Berlin))
    pool.map(est_features_run, run_files_Berlin)

    # l_missing = []
    # for run_file in run_files:
    #    raw_arr, dat, sfreq, line_noise, coord_list, coord_names = nm_IO.read_BIDS_data(run_file.path, PATH_BIDS)
    #    if coord_list is None:
    #        l_missing.append(run_file.filename)
