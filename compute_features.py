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
    if any(x in PATH_RUN for x in ["Berlin", "Pittsburgh", "Beijing"]):
        if "Berlin" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Berlin")
        elif "Beijing" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing"
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Beijing")
        elif "Pittsburgh" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
            PATH_OUT = os.path.join(PATH_OUT_BASE, "Pittsburgh")

        RUN_NAME = os.path.basename(PATH_RUN)[:-5]
        if os.path.exists(os.path.join(PATH_OUT, RUN_NAME)) is True:
            print("path exists")
            return
        (raw, data, sfreq, line_noise, coord_list, coord_names,) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg"
        )

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
            used_types=("ecog", "seeg", "dbs"),
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
                line_noise=line_noise
            )

            stream.run(
                data=data,
                out_path_root=PATH_OUT,
                folder_name=RUN_NAME,
            )
        except:
            print(f"could not run {RUN_NAME}")
    else:
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

        nm_channels = nm_define_nmchannels.set_channels(
            ch_names=ch_names,
            ch_types=ch_types,
            reference="default",
            bads=None,
            used_types=["ecog"],
            target_keywords=[
                "mov",
            ],
        )

        # read electrodes
        sub_name = RUN_NAME[:2]
        electrodes = (
            nm_IO.loadmat(
                os.path.join(
                    r"C:\Users\ICN_admin\Documents\Datasets\Washington\motor_basic\transform_mni",
                    sub_name + "_electrodes.mat",
                )
            )["mni_coord"]
            / 1000
        )  # transform into m

        stream = nm_stream_offline.Stream(
            settings=None,
            nm_channels=nm_channels,
            verbose=True,
        )

        stream.set_settings_fast_compute()
        stream.settings = set_settings(stream.settings)

        stream.init_stream(
            sfreq=sfreq,
            line_noise=60,
            coord_list=list(electrodes),
            coord_names=ch_names,
        )

        stream.nm_channels.loc[
            stream.nm_channels.query('type == "misc"').index, "target"
        ] = 1

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )


if __name__ == "__main__":
    # test runs:
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new"
    PATH_RUN = "C:\\Users\\ICN_admin\\Documents\\Datasets\\Berlin_new\\sub-012\\ses-EcogLfpMedOff01\\ieeg\\sub-012_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOff_run-1_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-007\ses-EcogLfpMedOff01\ieeg\sub-007_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-001\ses-EcogLfpMedOn01\ieeg\sub-001_ses-EcogLfpMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-004\ses-EcogLfpMedOff01\ieeg\sub-004_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr"

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new\sub-FOG006\ses-EphysMedOn01\ieeg\sub-FOG006_ses-EphysMedOn01_task-ButtonPressL_acq-StimOff_run-01_ieeg.vhdr"

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh\sub-001\ses-left\ieeg\sub-001_ses-left_task-force_run-0_ieeg.vhdr"

    PATH_DATA = (
        r"C:\Users\ICN_admin\Documents\Datasets\Washington\motor_basic\data"
    )
    run_files_Washington = [f for f in os.listdir(PATH_DATA) if "_mot_t_h" in f]
    PATH_RUN = os.path.join(PATH_DATA, run_files_Washington[0])
    #est_features_run(PATH_RUN)

    # collect all run files
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin"
    layout = BIDSLayout(PATH_BIDS)

    run_files_Berlin = layout.get(
        task=["SelfpacedRotationR", "SelfpacedRotationL", "SelfpacedForceWheel"],
        extension=".vhdr",
    )
    run_files_Berlin = [f.path for f in run_files_Berlin]
    #est_features_run(run_files_Berlin[10])

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing"
    layout = BIDSLayout(PATH_BIDS)
    run_files_Beijing = layout.get(
        task=["ButtonPressL", "ButtonPressR"], extension=".vhdr"
    )
    run_files_Beijing = [f.path for f in run_files_Beijing]

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
    layout = BIDSLayout(PATH_BIDS)
    run_files_Pittsburgh = layout.get(
        task=[
            "force",
        ],
        extension=".vhdr",
    )
    run_files_Pittsburgh = [f.path for f in run_files_Pittsburgh]

    PATH_DATA = (
        r"C:\Users\ICN_admin\Documents\Datasets\Washington\motor_basic\data"
    )
    run_files_Washington = [
        os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if "_mot_t_h" in f
    ]

    run_all = np.concatenate(
        [
            run_files_Berlin,
            run_files_Beijing,
            run_files_Pittsburgh,
            # run_files_Washington,
        ]
    )

    #for run_file in run_all:
    #    est_features_run(run_file)

    # run parallel Pool
    pool = Pool(processes=30)
    pool.map(est_features_run, run_all)

    # l_missing = []
    # for run_file in run_files:
    #    raw_arr, dat, sfreq, line_noise, coord_list, coord_names = nm_IO.read_BIDS_data(run_file.path, PATH_BIDS)
    #    if coord_list is None:
    #        l_missing.append(run_file.filename)
