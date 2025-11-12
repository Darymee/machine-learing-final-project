import warnings, os

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier, Pool

TARGET = "y"

# 1) Load
train = pd.read_csv("./final_proj_data.csv")
test = pd.read_csv("./final_proj_test.csv")
sub = pd.read_csv("./final_proj_sample_submission.csv")

# 2) Drop very-missing cols (as you did)
missing_pct = train.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 30].index
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# 3) Identify feature types
y = train[TARGET].values
X = train.drop(columns=[TARGET])
X_test = test.copy()

num_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Optional: add missing indicators (often helps a bit)
for c in num_feats + cat_feats:
    if X[c].isnull().any():
        ind_name = f"{c}__isna"
        X[ind_name] = X[c].isnull().astype(int)
        X_test[ind_name] = X_test[c].isnull().astype(int)
        num_feats.append(ind_name)  # indicator is numeric

# 4) CatBoost can handle NaNs; ensure categoricals are string dtype
for c in cat_feats:
    X[c] = X[c].astype("category").astype("str")
    X_test[c] = X_test[c].astype("category").astype("str")

# 5) CV setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# Choose the metric that matches the LB; change here if needed:
def metric(y_true, y_prob, name="balanced_accuracy"):
    if name == "accuracy":
        y_pred = (y_prob >= 0.5).astype(int)
        return accuracy_score(y_true, y_pred)
    if name == "f1":
        y_pred = (y_prob >= 0.5).astype(int)
        return f1_score(y_true, y_pred)
    if name == "roc_auc":
        return roc_auc_score(y_true, y_prob)
    if name == "balanced_accuracy":
        y_pred = (y_prob >= 0.5).astype(int)
        return balanced_accuracy_score(y_true, y_pred)


METRIC_NAME = "balanced_accuracy"  # <-- set this to the Kaggle LB metric if different

# 6) Reasonable param space; you can iterate a few variants manually
base_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",  # good guide even if LB metric differs
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=6.0,
    bagging_temperature=0.5,
    border_count=254,  # more splits for continuous vars
    random_strength=1.5,
    min_data_in_leaf=20,
    subsample=0.8,
    rsm=0.8,  # column sampling
    iterations=10000,  # rely on early stopping
    od_type="Iter",
    od_wait=200,
    verbose=False,
    random_seed=42,
    task_type="CPU",  # set "GPU" if you have one
)

oof_prob = np.zeros(len(X))
test_prob_folds = []

cat_idx = [X.columns.get_loc(c) for c in cat_feats if c in X.columns]

for fold, (tr, va) in enumerate(cv.split(X, y), 1):
    X_tr, X_va = X.iloc[tr], X.iloc[va]
    y_tr, y_va = y[tr], y[va]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)
    test_pool = Pool(X_test, cat_features=cat_idx)

    # You can try a few variants quickly (depth, l2, bagging_temperature)
    model = CatBoostClassifier(**base_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    # OOF probabilities
    oof_prob[va] = model.predict_proba(valid_pool)[:, 1]
    # Test fold probs
    test_prob_folds.append(model.predict_proba(test_pool)[:, 1])

    print(f"Fold {fold}: AUC={roc_auc_score(y_va, oof_prob[va]):.4f}")

# 7) Threshold tuning on OOF to maximize your chosen metric
best_thr, best_score = 0.5, -1
thr_grid = np.linspace(0.05, 0.95, 181)
for t in thr_grid:
    s = metric(
        y,
        (
            (oof_prob >= t).astype(int)
            if METRIC_NAME in ["accuracy", "f1", "balanced_accuracy"]
            else oof_prob
        ),
        METRIC_NAME,
    )
    if s > best_score:
        best_score, best_thr = s, t

print(f"OOF {METRIC_NAME}: {best_score:.5f} at threshold {best_thr:.3f}")
print(f"OOF ROC-AUC: {roc_auc_score(y, oof_prob):.5f}")

# 8) Fold-average test probabilities and apply tuned threshold
test_prob = np.mean(np.column_stack(test_prob_folds), axis=1)
test_pred = (test_prob >= best_thr).astype(int)

# 9) Save submission
out = sub.copy()
out[TARGET] = test_pred  # or probabilities if the competition expects probs
out.to_csv("final_submission1.csv", index=False)
print("Saved final_submission.csv")
