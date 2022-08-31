from py_neuromodulation import nm_cohortwrapper
import numpy as np

if __name__ == "__main__":

    cohort_runner = nm_cohortwrapper.CohortRunner(
        outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out",
        cohorts={"Berlin": "", "Washington": "", "Pittsburgh": "", "Beijing": ""},
    )
    gp_all = np.load(
        r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out\grid_point_all.npy",
        allow_pickle="TRUE",
    ).item()
    cohort_runner.rewrite_grid_point_all(gp_all, cohort_runner.outpath)
    cohort_runner.cohort_wrapper_read_all_grid_points(read_channels=False)
