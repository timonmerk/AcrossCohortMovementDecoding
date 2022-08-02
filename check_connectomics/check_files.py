import os

PATH_ROIs = (
    r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Washington\ROI Washington"
)

PATH_STRUCT = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Washington\structural connectivity"
PATH_FUNC = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Washington\functional connectivity"

# ROI Washingtonsub-bp_ROI_ECOG_0
# ROI Washingtonsub-bp_ROI_ECOG_0_func_seed_AvgR_Fz
f_missing = []
for ROI_f in os.listdir(PATH_ROIs):
    if len(list(filter(lambda k: ROI_f[:-4] in k, os.listdir(PATH_FUNC)))) == 0:
        f_missing.append(ROI_f)


# functional missing
ROI Washingtonsub-jm_ROI_ECOG_49.nii
ROI Washingtonsub-jm_ROI_ECOG_50.nii
ROI Washingtonsub-jm_ROI_ECOG_51.nii
ROI Washingtonsub-jm_ROI_ECOG_52.nii
ROI Washingtonsub-jm_ROI_ECOG_53.nii
ROI Washingtonsub-jm_ROI_ECOG_54.nii
ROI Washingtonsub-jm_ROI_ECOG_55.nii
ROI Washingtonsub-jm_ROI_ECOG_56.nii
ROI Washingtonsub-jm_ROI_ECOG_57.nii
ROI Washingtonsub-jm_ROI_ECOG_58.nii
ROI Washingtonsub-jm_ROI_ECOG_59.nii
ROI Washingtonsub-jm_ROI_ECOG_60.nii
ROI Washingtonsub-jm_ROI_ECOG_61.nii
ROI Washingtonsub-jm_ROI_ECOG_62.nii

# structural missing


ROI Washingtonsub-gf_ROI_ECOG_0.nii
ROI Washingtonsub-hh_ROI_ECOG_9.nii
ROI Washingtonsub-jf_ROI_ECOG_16.nii
ROI Washingtonsub-hl_ROI_ECOG_25.nii
ROI Washingtonsub-jm_ROI_ECOG_35.nii
ROI Washingtonsub-gc_ROI_ECOG_39.nii
ROI Washingtonsub-jm_ROI_ECOG_40.nii
ROI Washingtonsub-jm_ROI_ECOG_41.nii
ROI Washingtonsub-fp_ROI_ECOG_47.nii
ROI Washingtonsub-jm_ROI_ECOG_48.nii
ROI Washingtonsub-jm_ROI_ECOG_56.nii

