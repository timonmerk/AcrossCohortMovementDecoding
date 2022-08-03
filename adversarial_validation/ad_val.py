import numpy as np
import os
from sklearn import metrics, ensemble, model_selection
import pandas as pd
from py_neuromodulation import nm_across_patient_decoding
from matplotlib import pyplot as plt


# read ch_all
outpath = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"
ap = nm_across_patient_decoding.AcrossPatientRunner(outpath=outpath)

# read df with best ch
df_rmap = pd.read_csv("across patient running\\RMAP\\df_best_func_rmap_ch.csv")

# idx_name = np.load("idx_name.npy")
# cm = np.load("cm_ad_val.npy")

idx_name = []
y_sub = []
X_train = []
for cohort in ap.ch_all.keys():
    if cohort == "Washington":
        continue
    for sub in ap.ch_all[cohort].keys():
        ch_sel = df_rmap.query("cohort == @cohort and sub == @sub")["ch"].iloc[0]

        X_sub, y_test = ap.get_data_sub_ch(ap.ch_all, cohort, sub, ch_sel)

        class_name = f"{cohort}_{sub}"
        idx_name.append(class_name)
        y_sub.append(np.repeat(class_name, repeats=X_sub.shape[0]))
        X_train.append(X_sub)

X = np.concatenate(X_train)
y = np.concatenate(y_sub)

np.save("idx_name.npy", np.array(idx_name))

out_pr = model_selection.cross_val_predict(
    estimator=ensemble.RandomForestClassifier(n_jobs=50),
    X=X,
    y=y,
    cv=model_selection.StratifiedKFold(shuffle=True, n_splits=3),
)

cm = metrics.confusion_matrix(y, out_pr, normalize="true")

np.save("cm_ad_val.npy", cm)

disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=idx_name,
)
fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
disp.plot(ax=ax, include_values=False)
disp.ax_.set_title("Confusion Matrix Subject Prediction")

plt.xticks(rotation=90)
plt.savefig(
    os.path.join("figure", "cm_ad_validation.pdf"),
    bbox_inches="tight",
)

print("final")
