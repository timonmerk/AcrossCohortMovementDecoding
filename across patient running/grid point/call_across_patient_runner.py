from py_neuromodulation import nm_across_patient_decoding

if __name__ == "__main__":

    runner = nm_across_patient_decoding.AcrossPatientRunner(
        outpath=r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out",
        cohorts=["Pittsburgh", "Beijing", "Washington", "Berlin"],
        cv_method="NonShuffledTrainTestSplit",
        use_nested_cv=True,
    )

    runner.run_leave_nminus1_patient_out_across_cohorts()
