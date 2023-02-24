from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import shuffle
import pickle
import numpy as np
import os
import pandas as pd
import xgboost
from matplotlib import pyplot as plt

# select data from the best R-Map selected channel
OUTPATH = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"

ch_all = np.load(
    os.path.join(OUTPATH, "channel_all.npy"),
    allow_pickle="TRUE",
).item()

df = pd.read_csv("read_performances\\df_ch_performances.csv")
per_LM = []
pr_LM = []
per_XGB = []
pr_XGB = []

for time_range in np.array(np.rint(np.array([5, 10, 50, 100, 500, 1000])*3/2), dtype=int):

    print(time_range)
    dat_out = []
    label_out = []
    subs_ = []

    idx_ = 0
    for cohort in df["cohort"].unique():
        for sub in df.query("cohort == @cohort")["sub"].unique():
            df_sub = df.query("sub == @sub and cohort == @cohort")
            ch = str(df_sub.query('performance_test == performance_test.max()')["ch"].iloc[0])

            recs = [r for r in list(ch_all[cohort][sub][ch].keys()) if "StimOn" not in r]
            data = ch_all[cohort][sub][ch][recs[0]]["data"]
            label = ch_all[cohort][sub][ch][recs[0]]["label"]

            data, label = shuffle(data, label)

            idx_select = np.where(label == False)[0][:time_range]
            dat_out.append(data[idx_select, :])
            label_out.append(np.repeat(idx_, idx_select.shape[0]))
            subs_.append(f"{cohort}_{sub}")
            idx_ += 1

    X = np.concatenate(dat_out)
    y = np.concatenate(label_out)

    
    model_ = linear_model.LogisticRegression(n_jobs=35)

    pr = model_selection.cross_val_predict(
        estimator=model_,
        X=X,
        y=y,
        cv=model_selection.KFold(n_splits=3, shuffle=True)
    )

    per_LM.append(metrics.balanced_accuracy_score(y, pr))
    pr_LM.append(pr)

    model_ = xgboost.XGBClassifier(n_jobs=35)

    pr = model_selection.cross_val_predict(
        estimator=model_,
        X=X,
        y=y,
        cv=model_selection.StratifiedKFold(n_splits=3, shuffle=True)
    )

    per_XGB.append(metrics.balanced_accuracy_score(y, pr))
    pr_XGB.append(pr)

d = {}
d["per_LM"] = per_LM
d["per_XGB"] = per_XGB
d["pr_LM"] = pr_LM
d["pr_XGB"] = pr_XGB

with open('Patient_ID_Decoding\\per_different_times.p', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

PLT_ = False

if PLT_ is True:
    cm = metrics.confusion_matrix(y, pr, normalize="true")
    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(cm, aspect="auto")
    plt.title("Patient ID prediction LM ba=0.038, chance=0.017")
    plt.clim(0, 0.2)
    plt.xlabel("Patient")
    plt.ylabel("Patient")
    plt.colorbar()
    plt.savefig(
        "Patient_ID_Decoding\\confusion_matrix_patient_ID_decoding_LM.pdf",
        bbox_inches="tight",
    )
    plt.show()

    ConfusionMatrixDisplay.from_predictions(y, pr, normalize='true', include_values=False)
    plt.show()
