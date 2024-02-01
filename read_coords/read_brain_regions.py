# Import the AtlasBrowser class
from mni_to_atlas import AtlasBrowser

import sys

sys.path.append("read_coords")
import atlas_browser_plt_most_regions

import pandas as pd
import numpy as np

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
# coordinates = np.array([[-24, -53, 73],
#                        [-25, 20, 78]])

coordinates = pd.read_csv("read_performances\\df_ch_performances.csv")


# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates[["x", "y", "z"]].to_numpy(), plot=False)
coordinates["region"] = regions
coordinates["region_id"] = coordinates["region"].astype("category").cat.codes
# remove columns with region being "Undefined"
coordinates = coordinates[coordinates["region"] != "Undefined"]
# coordinates.to_csv("read_performances\\df_ch_performances_including_brain_regions.csv")

print(np.unique(np.array(regions)).shape[0])
# coordinates.groupby(["region", "cohort"]).size().reset_index().to_csv(
#    "read_performances\\grouped_regions_count_per_cohort.csv"
# )

coordinates.groupby(["region", "cohort"])["performance_test"].mean().reset_index()

# coordinates[coordinates["cohort"] != "Washington"].groupby(["region"])[
#    "performance_test"
# ].mean().reset_index().sort_values("performance_test", ascending=False).round(2).to_csv(
#    "read_performances\\grouped_regions_perf_per_cohort_wo_Washington.csv"
# )

# coordinates.groupby(["region"])["performance_test"].mean().reset_index().sort_values(
#    "performance_test", ascending=False
# ).round(2).to_csv("read_performances\\grouped_regions_perf_per_cohort_all.csv")

performances = coordinates.groupby(["region"])["performance_test"].mean().reset_index()
atlas_plt = atlas_browser_plt_most_regions.AtlasBrowser("AAL")
atlas_plt.find_regions(
    coordinates[["x", "y", "z"]].to_numpy()[:1, :], plot=True, performances=performances
)

# plot the regions
import seaborn as sns
from matplotlib import pyplot as plt

order = (
    coordinates.groupby(["region"])["performance_test"]
    .mean()
    .reset_index()
    .sort_values("performance_test", ascending=False)["region"]
)

sns.swarmplot(
    y="region", x="performance_test", data=coordinates, palette="viridis", order=order
)

sns.barplot(
    y="region",
    x="performance_test",
    data=coordinates,
    palette="viridis",
    order=order,
    estimator=np.nanmedian,
    errorbar=None,
)

atlas.find_regions(coordinates[["x", "y", "z"]].to_numpy()[:3, :], plot=True)
