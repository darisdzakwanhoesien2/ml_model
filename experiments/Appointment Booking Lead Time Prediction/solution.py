
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("dataset/public/train.csv")
test = pd.read_csv("dataset/public/test.csv")


TARGET = "booking_lead_time_days"
ID_COL = "appointment_id"

# ============================================================
# 2. TIME FEATURES
# ============================================================

def create_time_features(df):
    df = df.copy()
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])
    
    df["year"] = df["appointment_date"].dt.year
    df["month"] = df["appointment_date"].dt.month
    df["day_of_week"] = df["appointment_date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df

train = create_time_features(train)
test = create_time_features(test)

# ============================================================
# 3. PATIENT FEATURES (FIXED)
# ============================================================

# Sort for temporal safety
train = train.sort_values("appointment_date")

# Chronological (train only)
train["patient_past_mean"] = (
    train.groupby("patient_id")[TARGET]
    .expanding()
    .mean()
    .shift()
    .reset_index(level=0, drop=True)
)

train["patient_past_count"] = (
    train.groupby("patient_id")[TARGET]
    .cumcount()
)

global_mean = train[TARGET].mean()

train["patient_past_mean"] = train["patient_past_mean"].fillna(global_mean)
train["patient_past_count"] = train["patient_past_count"].fillna(0)

# ---- TEST SAFE VERSION ----
# Use global patient aggregates

patient_stats = (
    train.groupby("patient_id")[TARGET]
    .agg(["mean", "count"])
    .reset_index()
)

patient_stats.columns = [
    "patient_id",
    "patient_past_mean",
    "patient_past_count"
]

test = test.merge(patient_stats, on="patient_id", how="left")

test["patient_past_mean"] = test["patient_past_mean"].fillna(global_mean)
test["patient_past_count"] = test["patient_past_count"].fillna(0)

# ============================================================
# 4. TARGET ENCODING (SAFE)
# ============================================================

def target_encode_oof(train, test, col, target, n_splits=5):
    train_encoded = np.zeros(len(train))
    test_encoded = np.zeros(len(test))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for tr_idx, val_idx in kf.split(train):
        fold_train = train.iloc[tr_idx]
        fold_val = train.iloc[val_idx]
        
        means = fold_train.groupby(col)[target].mean()
        train_encoded[val_idx] = fold_val[col].map(means)
    
    full_means = train.groupby(col)[target].mean()
    test_encoded = test[col].map(full_means)
    
    train_encoded = pd.Series(train_encoded).fillna(global_mean)
    test_encoded = pd.Series(test_encoded).fillna(global_mean)
    
    return train_encoded, test_encoded

high_card_cols = ["city", "clinic_id", "specialty"]

for col in high_card_cols:
    train[f"{col}_te"], test[f"{col}_te"] = target_encode_oof(
        train, test, col, TARGET
    )

# ============================================================
# 5. ALIGN FEATURES (CRITICAL FIX)
# ============================================================

drop_cols = [TARGET, ID_COL, "appointment_date"]

train_features = [c for c in train.columns if c not in drop_cols]

# Ensure test has all columns
for col in train_features:
    if col not in test.columns:
        test[col] = 0

# Ensure same column order
X = train[train_features]
X_test = test[train_features]

y = np.log1p(train[TARGET])

# Convert object columns to category
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

# ============================================================
# 6. GROUP-AWARE CV (CITY HOLDOUT SAFE)
# ============================================================

n_splits = min(5, train["city"].nunique())
gkf = GroupKFold(n_splits=n_splits)

oof = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(
    gkf.split(X, y, groups=train["city"])
):
    
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.01,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(300)]
    )
    
    val_pred = np.expm1(model.predict(X_val))
    oof[val_idx] = val_pred
    
    test_preds += np.expm1(model.predict(X_test)) / n_splits
    
    fold_mae = mean_absolute_error(
        np.expm1(y_val), val_pred
    )
    
    print(f"Fold {fold+1} MAE: {fold_mae:.4f}")

overall_mae = mean_absolute_error(train[TARGET], oof)

print("\n===================================")
print(f"Overall CV MAE: {overall_mae:.4f}")
print("===================================")

# ============================================================
# 7. POST-PROCESSING
# ============================================================

test_preds = np.maximum(test_preds, 0)
test_preds = np.minimum(test_preds, 120)

# ============================================================
# 8. SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "appointment_id": test[ID_COL],
    "booking_lead_time_days": test_preds
})

assert submission.isna().sum().sum() == 0
assert submission.shape[1] == 2

submission.to_csv("./working/submission.csv", index=False)

print("Submission file created successfully.")