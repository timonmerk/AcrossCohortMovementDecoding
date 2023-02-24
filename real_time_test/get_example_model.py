import xgboost
from sklearn import linear_model

import pandas as pd

df = pd.read_csv(r"C:\Users\ICN_admin\Documents\PAPERD~1\ACROSS~2\FEATUR~1\Berlin\SUDFDF~1\SUB-00~1.CSV")

X = df[[f for f in df.columns if "ECOG_L_1_" in f]]
y = df["SQUARED_INTERPOLATED_EMG"]

model_LM = linear_model.LogisticRegression().fit(X, y)
model_XGBOOST = xgboost.XGBClassifier().fit(X, y)

import pickle
# now you can save it to a file
with open('lm_model.pkl', 'wb') as f:
    pickle.dump(model_LM, f)

with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(model_XGBOOST, f)
