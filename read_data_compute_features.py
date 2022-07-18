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
        settings["features"]["fooof"] = True
        settings["features"]["return_raw"] = True
        settings["features"]["raw_hjorth"] = True
        settings["features"]["sharpwave_analysis"] = True
        settings["features"]["nolds"] = True
        settings["features"]["bursts"] = True
        settings["features"]["coherence"] = False

        settings["preprocessing"]["re_referencing"] = True
        settings["preprocessing"]["notch_filter"] = True
        settings["preprocessing"]["raw_resampling"] = True
        settings["preprocessing"]["preprocessing_order"] = [
            "re_referencing",
            "raw_resampling",
            "notch_filter",
        ]

        settings["postprocessing"]["feature_normalization"] = True
        settings["postprocessing"]["project_cortex"] = True
        settings["postprocessing"]["project_subcortex"] = False

        settings["fooof"]["periodic"]["center_frequency"] = False
        settings["fooof"]["periodic"]["band_width"] = False
        settings["fooof"]["periodic"]["height_over_ap"] = False

        for key in list(
            settings["sharpwave_analysis_settings"]["sharpwave_features"].keys()
        ):
            settings["sharpwave_analysis_settings"]["sharpwave_features"][
                key
            ] = True
        settings["sharpwave_analysis_settings"]["sharpwave_features"][
            "peak_left"
        ] = False
        settings["sharpwave_analysis_settings"]["sharpwave_features"][
            "peak_right"
        ] = False
        settings["sharpwave_analysis_settings"]["sharpwave_features"][
            "trough"
        ] = False
        settings["sharpwave_analysis_settings"][
            "apply_estimator_between_peaks_and_troughs"
        ] = True

        settings["sharpwave_analysis_settings"]["estimator"]["max"] = [
            "prominence",
            "sharpness",
        ]
        settings["sharpwave_analysis_settings"]["estimator"]["mean"] = [
            "width",
            "interval",
            "decay_time",
            "rise_time",
            "rise_steepness",
            "decay_steepness",
            "slope_ratio",
        ]

        # settings["coherence"]["channels"] = [
        #    ["Cg25L01", "Cg25L03"],
        #    ["Cg25R01", "Cg25R03"],
        #    ["Cg25L01", "Cg25R03"],
        #    ["Cg25R01", "Cg25L03"],
        # ]

        # settings["coherence"]["frequency_bands"] = ["high beta", "low gamma"]
        # settings["coherence"]["method"]["coh"] = True
        # settings["coherence"]["method"]["icoh"] = True

        settings["nolds_features"]["sample_entropy"] = False
        settings["nolds_features"]["correlation_dimension"] = False
        settings["nolds_features"]["lyapunov_exponent"] = False
        settings["nolds_features"]["hurst_exponent"] = True
        settings["nolds_features"]["detrended_fluctutaion_analysis"] = False
        settings["nolds_features"]["data"]["raw"] = True
        settings["nolds_features"]["data"]["frequency_bands"] = [
            "low beta",
            "high beta",
            "low gamma",
            "HFA",
        ]

        settings["features"]["fft"] = True
        settings["features"]["fooof"] = False
        settings["features"]["return_raw"] = False
        settings["features"]["raw_hjorth"] = False
        settings["features"]["sharpwave_analysis"] = False
        settings["features"]["nolds"] = False
        settings["features"]["bursts"] = False
        settings["features"]["coherence"] = False

        return settings

    if any(x in PATH_RUN for x in ["Berlin", "Pittsburgh", "Beijing"]):
        if "Berlin" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new"
            PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out\Berlin" 
        elif "Beijing" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new"
            PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out\Beijing"
        elif "Pittsburgh" in PATH_RUN:
            PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
            PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out\Pittsburgh"
        RUN_NAME = os.path.basename(PATH_RUN)[:-5]
        (
            raw,
            data,
            sfreq,
            line_noise,
            coord_list,
            coord_names,
        ) = nm_IO.read_BIDS_data(
            PATH_RUN=PATH_RUN, BIDS_PATH=PATH_BIDS, datatype="ieeg"
        )

        # cut for Berlin sub012 the last ECoG channel, due to None coordinates
        #if "Berlin" in PATH_RUN and "sub-012" in PATH_RUN:
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

        stream = nm.Stream(
            settings=None,
            nm_channels=nm_channels,
            path_grids=None,
            verbose=True,
        )

        stream.reset_settings()
        stream.settings = set_settings(stream.settings)

        stream.init_stream(
            sfreq=sfreq,
            line_noise=line_noise,
            coord_list=coord_list,
            coord_names=coord_names,
        )

        stream.run(
            data=data,
            out_path_root=PATH_OUT,
            folder_name=RUN_NAME,
        )
    else:
        PATH_OUT = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out\Washington"
        RUN_NAME = os.path.basename(PATH_RUN[:-4])  # cut .mat

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
                    r"C:\Users\ICN_admin\Documents\Datasets\Data Kai Miller\motor_basic\transform_mni",
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
    PATH_RUN = 'C:\\Users\\ICN_admin\\Documents\\Datasets\\Berlin_new\\sub-012\\ses-EcogLfpMedOff01\\ieeg\\sub-012_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOff_run-1_ieeg.vhdr'
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-007\ses-EcogLfpMedOff01\ieeg\sub-007_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOn_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-001\ses-EcogLfpMedOn01\ieeg\sub-001_ses-EcogLfpMedOn01_task-SelfpacedForceWheel_acq-StimOff_run-01_ieeg.vhdr"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new\sub-004\ses-EcogLfpMedOff01\ieeg\sub-004_ses-EcogLfpMedOff01_task-SelfpacedRotationL_acq-StimOff_run-01_ieeg.vhdr"

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new\sub-FOG006\ses-EphysMedOn01\ieeg\sub-FOG006_ses-EphysMedOn01_task-ButtonPressL_acq-StimOff_run-01_ieeg.vhdr"

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
    PATH_RUN = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh\sub-001\ses-left\ieeg\sub-001_ses-left_task-force_run-0_ieeg.vhdr"

    PATH_DATA = r"C:\Users\ICN_admin\Documents\Datasets\Data Kai Miller\motor_basic\data"
    run_files_Washington = [f for f in os.listdir(PATH_DATA) if "_mot_t_h" in f]
    PATH_RUN = os.path.join(PATH_DATA, run_files_Washington[0])
    #est_features_run(PATH_RUN)

    # collect all run files
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new"
    layout = BIDSLayout(PATH_BIDS)

    run_files_Berlin = layout.get(task=['SelfpacedRotationR', 'SelfpacedRotationL', 'SelfpacedForceWheel'], extension='.vhdr')
    run_files_Berlin = [f.path for f in run_files_Berlin]

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Beijing_new"
    layout = BIDSLayout(PATH_BIDS)
    run_files_Beijing = layout.get(task=['ButtonPressL', 'ButtonPressR'], extension='.vhdr')
    run_files_Beijing = [f.path for f in run_files_Beijing]

    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Pittsburgh"
    layout = BIDSLayout(PATH_BIDS)
    run_files_Pittsburgh = layout.get(task=['force',], extension='.vhdr')
    run_files_Pittsburgh = [f.path for f in run_files_Pittsburgh]

    PATH_DATA = r"C:\Users\ICN_admin\Documents\Datasets\Data Kai Miller\motor_basic\data"
    run_files_Washington = [f for f in os.listdir(PATH_DATA) if "_mot_t_h" in f]

    run_all = np.concatenate([run_files_Berlin, run_files_Beijing, run_files_Pittsburgh, run_files_Washington])

    # run parallel Pool
    pool = Pool(processes=50)
    pool.map(est_features_run, run_all)

    #for run_file in run_all:
    #    est_features_run(run_file)

    # run LM Decoder

    #l_missing = [] 
    #for run_file in run_files:
    #    raw_arr, dat, sfreq, line_noise, coord_list, coord_names = nm_IO.read_BIDS_data(run_file.path, PATH_BIDS)
    #    if coord_list is None:
    #        l_missing.append(run_file.filename)

