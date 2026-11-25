"""Train the phishing model on Dataset_expanded.csv and write Final_model.pkl + model_scaler.pkl

The train/test split is GROUPED BY REGISTERED DOMAIN, so the test set only contains
domains the model never saw in training (a much more honest estimate for "new links").

Run:  python train.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from feature_extraction import FEATURE_ORDER, registered_domain

SEED = 42
df = pd.read_csv("Dataset_expanded.csv")
print("rows:", df.shape)

groups = df.url.map(registered_domain)
X = df[FEATURE_ORDER]
y = df.label

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
print("train/test:", len(tr), len(te), "| shared domains:", len(set(groups.iloc[tr]) & set(groups.iloc[te])))

scaler = StandardScaler().fit(X.iloc[tr])
Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])

model = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=30,
    l2_regularization=1.0, class_weight="balanced", early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=25, random_state=SEED,
)
model.fit(Xtr, y.iloc[tr])
print("trees used:", model.n_iter_)

proba = model.predict_proba(Xte)[:, 1]
THRESHOLD = 0.5


def report(name, mask):
    yt, p = y.iloc[te][mask].values, proba[mask]
    pr = (p >= THRESHOLD).astype(int)
    print(f"{name:<22} n={mask.sum():>6}  acc={(pr == yt).mean():.4f}  prec={precision_score(yt, pr):.4f}  "
          f"rec={recall_score(yt, pr):.4f}  f1={f1_score(yt, pr):.4f}  auc={roc_auc_score(yt, p):.4f}")


src = df.source.iloc[te].values
report("ALL (unseen domains)", np.ones(len(te), bool))
report("real data only", src == "real")
report("synthetic only", src == "synthetic")
print(classification_report(y.iloc[te], (proba >= THRESHOLD).astype(int), target_names=["legit", "phishing"], digits=4))

# final model: refit on 100% of the data with the same settings
scaler = StandardScaler().fit(X)
model.set_params(early_stopping=False, max_iter=int(model.n_iter_))
model.fit(scaler.transform(X), y)

with open("Final_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model_scaler.pkl", "wb") as f:
    pickle.dump({"scaler": scaler, "feature order": FEATURE_ORDER, "threshold": THRESHOLD}, f)
print("saved Final_model.pkl, model_scaler.pkl")
