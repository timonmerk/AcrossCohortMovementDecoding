import os
import mne_bids
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from bids import BIDSLayout
from py_neuromodulation import (
    nm_across_patient_decoding,
    nm_analysis,
    nm_decode,
    nm_plots,
)
from sklearn import linear_model, metrics
from multiprocessing import Pool

df = pd.read_csv("read_performances\\df_grid_point_performances.csv")

print("hallo")
