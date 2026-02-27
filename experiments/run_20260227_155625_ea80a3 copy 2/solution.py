import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

TARGET = "accepted_flag"
ID_COL = "id"

# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Drop extremely sparse column if exists
if "vehicle_class" in train.columns:
    missing_ratio = train["vehicle_class"].isnull().mean()
    if missing_ratio > 0.9:
        train = train.drop(columns=["vehicle_class"])
        test = test.drop(columns=["vehicle_class"])

# Fill remaining missing categorical values
train = train.fillna("Unknown")
test = test.fillna("Unknown")

# ============================================================
# 3. FEATURE SETUP
# ============================================================

features = [c for c in train.columns if c not in [TARGET, ID_COL]]

categorical_cols = train[features].select_dtypes(include=["object"]).columns.tolist()

# Convert to category for LightGBM
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")

# ============================================================
# 4. STRATIFIED CROSS VALIDATION
# ============================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = np.zeros(len(train))
test_probs = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):

    X_tr = train.iloc[train_idx][features]
    y_tr = train.iloc[train_idx][TARGET]

    X_val = train.iloc[val_idx][features]
    y_val = train.iloc[val_idx][TARGET]

    base_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    base_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(200)
        ]
    )

    # Probability calibration (Platt scaling)
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val, y_val)

    oof_probs[val_idx] = calibrated_model.predict_proba(X_val)[:, 1]
    test_probs += calibrated_model.predict_proba(test[features])[:, 1] / 5

    print(f"Fold {fold+1} complete.")

# ============================================================
# 5. LOCAL METRIC EVALUATION
# ============================================================

def precision_at_budget(y_true, y_score, budget_fraction=0.9):
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df = df.sort_values("score", ascending=False)
    k = int(np.floor(budget_fraction * len(df)))
    top_k = df.iloc[:k]
    return top_k["y"].sum() / k

precision_90 = precision_at_budget(train[TARGET], oof_probs, 0.9)
brier = brier_score_loss(train[TARGET], oof_probs)
calibration_score = 1 - brier

final_score = 0.9 * precision_90 + 0.1 * calibration_score

print("\n====================================")
print(f"Precision@90%: {precision_90:.5f}")
print(f"Brier Score: {brier:.5f}")
print(f"Calibration Score: {calibration_score:.5f}")
print(f"Final Score: {final_score:.5f}")
print("====================================")

# ============================================================
# 6. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "id": test[ID_COL],
    "score": test_probs
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully.")