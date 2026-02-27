# ============================================================
# POWER PLANT UNDERPERFORMANCE PREDICTION
# COMPLETE SOLUTION PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "underperforming"
ID_COL = "id"

# ============================================================
# 2. BASIC FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):
    df = df.copy()

    # Capacity skew handling
    df["capacity_log2"] = np.log1p(df["capacity_mw"])

    # Interaction refinements
    df["age_capacity_ratio"] = df["plant_age"] / (df["capacity_log_mw"] + 1e-6)
    df["geo_interaction"] = df["latitude"] * df["longitude"]

    # Distance from equator proxy
    df["distance_from_equator"] = np.abs(df["latitude"])

    return df

train = feature_engineering(train)
test = feature_engineering(test)

# ============================================================
# 3. DEFINE FEATURES
# ============================================================

features = [c for c in train.columns if c not in [TARGET, ID_COL]]

categorical_cols = [
    "fuel_group",
    "primary_fuel",
    "other_fuel1",
    "owner_bucket",
    "capacity_band",
    "lat_band",
    "lon_band"
]

# Convert to category dtype for LightGBM
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# ============================================================
# 4. STRATIFIED CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):

    X_tr = train.iloc[train_idx][features]
    y_tr = train.iloc[train_idx][TARGET]

    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx][TARGET]

    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )

    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(test[features])[:, 1] / 5

    fold_auc = roc_auc_score(y_val, oof[val_idx])
    fold_ap = average_precision_score(y_val, oof[val_idx])

    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.5f}")
    print(f"Fold {fold+1} AP: {fold_ap:.5f}")

# ============================================================
# 5. OVERALL METRICS
# ============================================================

overall_auc = roc_auc_score(train[TARGET], oof)
overall_ap = average_precision_score(train[TARGET], oof)

final_score = 0.7 * overall_auc + 0.3 * overall_ap

print("\n===================================")
print(f"Overall ROC-AUC: {overall_auc:.5f}")
print(f"Overall AP: {overall_ap:.5f}")
print(f"Composite Score: {final_score:.5f}")
print("===================================")

# ============================================================
# 6. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "id": test[ID_COL],
    "underperforming": test_preds
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully.")