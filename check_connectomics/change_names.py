import os


# smooth structural conn!

# Berlin currently: Berlin_sub-8_ROI_ECOG_L_1_SMC_AT

# Washington currently: ROI Washingtonsub-bp_ROI_ECOG_0
# also only Washington is not smoothed

# Washington:
# standard: Pittsburgh_000_ROI_ECOG_RIGHT_0
PATH_WASHINGTON = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Washington\structural connectivity"
for f in os.listdir(PATH_WASHINGTON):
    if f == "non smoothed":
        continue
    print(f)
    sub = f[f.find("sub-") + 4 : f.find("_ROI")]
    suffix = f[f.find("ECOG") :]
    # sub_ = f"{sub:03}"
    path_change = f"sWashington_{sub}_ROI_{suffix}"

    # path_change = f"s{f}"
    os.rename(
        os.path.join(PATH_WASHINGTON, f), os.path.join(PATH_WASHINGTON, path_change)
    )

# Berlin:

PATH_BERLIN = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Berlin\functional connectivity_008_014"
PATH_BERLIN = r"C:\Users\ICN_admin\Documents\Datasets\Connectomes\Berlin\structural connectivity_008_014"
for f in os.listdir(PATH_BERLIN):
    if f == "Berlin_010_ROI_ECOG_R_1_SMC_AT_func_seed_AvgR.nii":
        continue
    if f == "non_smoothed":
        continue
    print(f)
    sub = int(f[f.find("sub-") + 4 : f.find("_ROI")])
    suffix = f[f.find("ECOG") :]
    sub_ = f"{sub:03}"
    path_change = f"Berlin_{sub_}_ROI_{suffix}"

    path_change = f"s{f}"
    os.rename(os.path.join(PATH_BERLIN, f), os.path.join(PATH_BERLIN, path_change))
