from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import shuffle
import pickle
import numpy as np
import os
import pandas as pd
import xgboost
from matplotlib import pyplot as plt

with open('Patient_ID_Decoding\\per_different_times.p', 'rb') as handle:
    d = pickle.load(handle)

time_train = np.round(np.array([5, 10, 50, 100, 500, 1000])*10/1000, 2)

plt.figure(figsize=(4, 3), dpi=300)
plt.plot(time_train, d["per_LM"], label="LM")
plt.plot(time_train, d["per_XGB"], label="XGB")
plt.xlabel("Time training [s]")
plt.ylabel("Balanced accuracy")
plt.title("Cross Validation performances\nPatient ID decoding")
plt.axhline(y=1/56, color='gray', linestyle='--', label="chance")
plt.ylim(-0.02, 0.4)
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(
    "Patient_ID_Decoding\\TimeRangeDecoding_log.pdf",
    bbox_inches="tight",
)


