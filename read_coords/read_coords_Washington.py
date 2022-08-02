from py_neuromodulation import nm_IO
import os
import pandas as pd
from bids import BIDSLayout
import numpy as np

if __name__ == "__main__":

    # read Berlin sub 008 till 012
    PATH_BIDS = r"C:\Users\ICN_admin\Documents\Datasets\Berlin_new"
    layout = BIDSLayout(PATH_BIDS)

    run_files_Berlin = layout.get(
        task=["SelfpacedRotationR", "SelfpacedRotationL", "SelfpacedForceWheel"],
        extension=".vhdr",
    )
    run_files_Berlin = list(
        np.array([f.path for f in run_files_Berlin[26:]])[[0, 3, 6, 8, 11]]
    )  # from sub008
    df = pd.DataFrame()
    for run_file in run_files_Berlin:
        _, _, _, _, coord_list, coord_names = nm_IO.read_BIDS_data(run_file, PATH_BIDS)
        for idx, coord_name in enumerate(coord_names):
            if "ECOG" in coord_name:
                d = {}
                d["ch"] = coord_name
                d["x"] = coord_list[idx][0] * 1000
                d["y"] = coord_list[idx][1] * 1000
                d["z"] = coord_list[idx][2] * 1000
                d["sub"] = os.path.basename(run_file)[4:7]
                df = df.append(d, ignore_index=True)
    df.drop(29)
    df.to_csv("berlin_electrodes.tsv")
    # read coords Washington

    PATH_DATA = (
        r"C:\Users\ICN_admin\Documents\Datasets\Data Kai Miller\motor_basic\data"
    )
    run_files_Washington = [
        os.path.join(PATH_DATA, f) for f in os.listdir(PATH_DATA) if "_mot_t_h" in f
    ]

    l_ = []

    for PATH_RUN in run_files_Washington:
        RUN_NAME = os.path.basename(PATH_RUN[:-4])
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
        ch_names = [f"ECOG_{i}" for i in range(electrodes.shape[0])]
        df = pd.DataFrame()
        df["ch"] = ch_names
        df["x"] = electrodes[:, 0] * 1000
        df["y"] = electrodes[:, 1] * 1000
        df["z"] = electrodes[:, 2] * 1000
        df["sub"] = sub_name
        l_.append(df)
    pd.concat(l_).to_csv("washington_electrodes.csv")
