from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils import shuffle
import pickle
import numpy as np
import os
import random
import pandas as pd
import xgboost
from matplotlib import pyplot as plt
from py_neuromodulation import nm_analysis

def write_example_RNS_data(time_range = 13):
    #  13 seconds for 10 s training phase

    dat_out = []
    subs_ = []

    PATH_FEATURES_EPILEPSY = r"C:\Users\ICN_admin\OneDrive - Charité - Universitätsmedizin Berlin\Dokumente\Decoding toolbox\epilepsy\predictions_sweep\folder_features"
    RNS_subjects = ["1090", "1440", "1529", "1534", "1603", "1836", "6806", "8973", "9536"]
    files_ = [i for i in os.listdir(PATH_FEATURES_EPILEPSY) if i.startswith("RNS")]

    for sub in RNS_subjects:
        r = [f for f in files_ if sub in f][0]
        dat = os.path.join(PATH_FEATURES_EPILEPSY, r)
        with open(dat, 'rb') as handle:
            data = pickle.load(handle)

            # check for the first run that does NOT have seizure activity
            for episode in list(data.keys()):
                if data[episode]["sz"].sum() != 0:  # select only non-ictal activity
                    continue
                f_select = [c for c in data[episode].columns if "fft" in c]
                dat_out.append(data[episode][f_select].iloc[:time_range])
                break
            
            subs_.append(f"RNS_{sub}")

    d_eps = {}
    for sub_idx, sub in enumerate(subs_):
        d_eps[sub] = {}
        for ch in ["ch1", "ch2", "ch3", "ch4"]:
            features_ch = [f for f in dat_out[sub_idx].columns if ch in f]
            d_eps[sub][ch] = dat_out[sub_idx][features_ch]

    with open('Patient_ID_Decoding\\RNS_example_data.p', 'wb') as handle:
        pickle.dump(d_eps, handle, protocol=pickle.HIGHEST_PROTOCOL)

def write_example_data_movement_cohorts(time_range: int = 133):
    OUTPATH = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\AcrossCohortMovementDecoding\features_out_fft"

    ch_all = np.load(
        os.path.join(OUTPATH, "channel_all.npy"),
        allow_pickle="TRUE",
    ).item()

    df = pd.read_csv("read_performances\\df_ch_performances.csv")

    # idea decode:
    # 1. ch
    # 2. subject
    # 3. disease
    # 4. stimulation?

    # first check with all channels, then limit down to random one?
    # limit down to 10s per subject

    dat_out = {}

    for cohort in df["cohort"].unique():
        dat_out[cohort] = {}
        for sub in df.query("cohort == @cohort")["sub"].unique():
            df_sub = df.query("sub == @sub and cohort == @cohort")
            dat_out[cohort][sub] = {}

            for ch in list(df_sub["ch"].unique()):

                recs = [r for r in list(ch_all[cohort][sub][ch].keys()) if "StimOn" not in r]
                data = ch_all[cohort][sub][ch][recs[0]]["data"]
                label = ch_all[cohort][sub][ch][recs[0]]["label"]

                data, label = shuffle(data, label)

                idx_select = np.where(label == False)[0][:time_range]
                dat_out[cohort][sub][ch] = data[idx_select, :]

    with open('Patient_ID_Decoding\\Movement_cohorts_data.p', 'wb') as handle:
        pickle.dump(dat_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

def write_depression_example_data(time_range: int=130):
    PATH_FEATURES = r"C:\Users\ICN_admin\Documents\Paper Decoding Toolbox\TRD Analysis\features_computed"
    subjects = [
        f for f in os.listdir(PATH_FEATURES) if f.startswith("effspm8") and "KSC" not in f
    ]
    dat_out = {}

    for sub in subjects:
        dat_out[sub] = {}

        analyzer = nm_analysis.Feature_Reader(
            feature_dir=PATH_FEATURES,
            feature_file=sub,
            binarize_label=False,
        )
        ch_names = [c[:-len("_RawHjorth_Activity")] for c in analyzer.feature_arr.columns if "_RawHjorth_Activity" in c]

        for ch in ch_names:
            
            rest_ = analyzer.feature_arr.query("ALL == 0")
            idx_select = random.sample(range(rest_.shape[0]), time_range)

            ch_dat = analyzer.feature_arr[[c for c in analyzer.feature_arr.columns if ch in c and "fft" in c]]
            # select now only rest data (pre-stim)
            dat_out[sub][ch] = ch_dat.iloc[idx_select, :]

    with open('Patient_ID_Decoding\\Depression_cohorts_data.p', 'wb') as handle:
        pickle.dump(dat_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

WRITE_DAT = False

if WRITE_DAT == True:
    write_depression_example_data(time_range=130)
    write_example_data_movement_cohorts(time_range=130)
    write_example_RNS_data(time_range=13)

# read now all data and collect all common features
# run than channel wise prediction

with open('Patient_ID_Decoding\\RNS_example_data.p', 'rb') as handle:
    eps = pickle.load(handle)  # theta, alpha, low beta, high beta, low gamma, broadband

with open('Patient_ID_Decoding\\Depression_cohorts_data.p', 'rb') as handle:
    dep = pickle.load(handle)  # theta, alpha, low beta, high beta, low gamma, high gamma, HFA

with open('Patient_ID_Decoding\\Movement_cohorts_data.p', 'rb') as handle:
    mov = pickle.load(handle)  # theta, alpha, low beta, high beta, low gamma, high gamma, HFA

# common: theta, alpha, low beta, high beta, low gamma
# validation: channels: cohort_sub_ch

X = []
y = []
idx = 0
sub_label = []
INCLUDE_EPS = False
if INCLUDE_EPS == True:
    for sub in eps.keys():
        ch_list = list(eps[sub].keys())
        ch_list = [ch_list[np.random.choice(len(ch_list), 1, replace=False)[0]]] # take a random channel
        for ch in ch_list:  
            dat_add = np.array(eps[sub][ch].iloc[:, :5])
            X.append(dat_add)
            y.append(np.repeat(idx, dat_add.shape[0]))
            idx += 1
            sub_label.append(f"Epilepsy_{sub[5:]}_{ch}")

for sub in dep.keys():
    ch_list = list(dep[sub].keys())
    ch_list = [ch_list[np.random.choice(len(ch_list), 1, replace=False)[0]]] # take a random channel
    for ch in ch_list:
        dat_add = np.array(dep[sub][ch].iloc[:, :5])
        X.append(dat_add)
        y.append(np.repeat(idx, dat_add.shape[0]))
        idx += 1
        sub_label.append(f"Depression_{sub}_{ch}")

for cohort in mov.keys():
    for sub in mov[cohort].keys():
        ch_list = list(mov[cohort][sub].keys())
        ch_list = [ch_list[np.random.choice(len(ch_list), 1, replace=False)[0]]] # take a random channel
        for ch in ch_list:
            dat_add = mov[cohort][sub][ch][:, :5]
            X.append(dat_add)
            y.append(np.repeat(idx, dat_add.shape[0]))
            idx += 1
            sub_label.append(f"Movmenet_{cohort}_{sub}_{ch}")

X_ = np.array(np.nan_to_num(np.concatenate(X)), dtype= np.float16)
y_ = np.array(np.concatenate(y), dtype=np.float16)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_, y_, test_size=0.33,
                                                    random_state=42, stratify=y_)

#  tree_method="gpu_hist", gpu_id=0, 
model = xgboost.XGBClassifier(verbose=10, n_jobs=-1)

model.fit(X_train, y_train, verbose=True)

y_pr = model.predict(X_test)

ba = np.round(metrics.balanced_accuracy_score(y_test, y_pr), 4)

cm = metrics.confusion_matrix(y_test, y_pr, normalize="true")
plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(cm, aspect="auto")
plt.title(f"Patient ch prediction XGBoost ba={ba}")
plt.clim(0, 0.2)
plt.xlabel("ch")
plt.ylabel("ch")
plt.colorbar()
plt.tight_layout()

#model = linear_model.LogisticRegression(n_jobs=50, solver="lbfgs")

pr = model_selection.cross_val_predict(
    estimator=model,
    X=X_,
    y=y_,
    cv=model_selection.StratifiedKFold(n_splits=3, shuffle=True),
    n_jobs=50
)


params = {
         'tree_method': 'gpu_hist' # Use GPU accelerated algorithm
         }

# Convert input data from numpy to XGBoost format
dtrain = xgboost.DMatrix(X_, label=y_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=0.75,
                                                    random_state=42)

xgb_cv = xgboost.cv(params, dtrain, nfold=3, 
stratified=True, folds=None, metrics=(), obj=None, feval=None, 
maximize=False, early_stopping_rounds=None, fpreproc=None, 
as_pandas=True, verbose_eval=True, show_stdv=True, seed=0, 
callbacks=None, shuffle=True)


xgb.train(param, dtrain, num_round, evals=[(dtest, 'test')], evals_result=gpu_res)



ba = np.round(metrics.balanced_accuracy_score(y_, pr), 4)

cm = metrics.confusion_matrix(y_, pr, normalize="true")
plt.figure(figsize=(5, 5), dpi=300)
plt.imshow(cm, aspect="auto")
plt.title(f"Patient ch prediction XGBoost ba={ba}")
plt.clim(0, 0.001)
plt.xlabel("ch")
plt.ylabel("ch")
plt.colorbar()

plt.savefig(
    "Patient_ID_Decoding\\confusion_matrix_patient_ID_decoding_LM.pdf",
    bbox_inches="tight",
)
plt.show()

