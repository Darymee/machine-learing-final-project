import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TARGET = "y"

# 1) Load data
train = pd.read_csv("./final_proj_data.csv")
test = pd.read_csv("./final_proj_test.csv")
submission = pd.read_csv("./final_proj_sample_submission.csv")

# 2) Drop columns with too many missing
missing_pct = train.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 30].index
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# 3) Feature types
target = "y"
num_feats = (
    train.select_dtypes(include=["int64", "float64"]).drop(columns=target).columns
)
cat_feats = train.select_dtypes(include=["object"]).columns

# 4) Preprocessing
num_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Используем параметр `sparse=False`, чтобы работать и в более старых версиях sklearn
cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    [("num", num_pipeline, num_feats), ("cat", cat_pipeline, cat_feats)]
)

# 5) Class weights
classes = np.unique(train[target])
weights = compute_class_weight(
    class_weight="balanced", classes=classes, y=train[target]
)
class_weights = dict(zip(classes, weights))

# 6) Initial pipeline (без SMOTE)
initial_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "clf",
            CatBoostClassifier(random_state=42, verbose=0, class_weights=class_weights),
        ),
    ]
)

# 7) CV before tuning
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(
    initial_pipeline,
    train.drop(columns=target),
    train[target],
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
)
print("Balanced accuracy before tuning:", scores, f"Mean: {scores.mean():.4f}")

# 8) Hyperparameter tuning
param_dist = {
    "clf__iterations": [100, 200, 300],
    "clf__depth": [4, 6, 8],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__l2_leaf_reg": [1, 3, 5, 7],
}

search_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "clf",
            CatBoostClassifier(random_state=42, verbose=0, class_weights=class_weights),
        ),
    ]
)

search = RandomizedSearchCV(
    estimator=search_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
    random_state=42,
)
search.fit(train.drop(columns=target), train[target])

best_model = search.best_estimator_
scores_tuned = cross_val_score(
    best_model,
    train.drop(columns=target),
    train[target],
    scoring="balanced_accuracy",
    cv=cv,
    n_jobs=-1,
)
print(
    "Balanced accuracy after tuning:", scores_tuned, f"Mean: {scores_tuned.mean():.4f}"
)
print("Best params:", search.best_params_)

# 9) Fit on full train and predict test
best_model.fit(train.drop(columns=target), train[target])
preds = best_model.predict(test)

# 10) Save submission
submission["y"] = preds
output_file = "final_submission.csv"
submission.to_csv(output_file, index=False)
print(f"Submission saved to {output_file}")
