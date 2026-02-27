import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = [c for c in train.columns if c.startswith("f")]

# ============================================================
# 2. CAUSAL MISSING VALUE IMPUTATION (NO LOOKAHEAD)
# ============================================================

def causal_impute(df):
    df = df.copy()
    df[features] = df[features].ffill()
    df[features] = df[features].fillna(0)
    return df

train = causal_impute(train)
test = causal_impute(test)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def create_features(df):
    df = df.copy()

    # Lag features
    for lag in [1, 2, 3, 5]:
        for f in features:
            df[f"{f}_lag{lag}"] = df[f].shift(lag)

    # Rolling volatility
    for f in features:
        df[f"{f}_roll_std5"] = df[f].rolling(5).std()
        df[f"{f}_roll_std20"] = df[f].rolling(20).std()

    # Cross-sectional stats
    df["cross_vol"] = df[features].std(axis=1)
    df["cross_mean"] = df[features].mean(axis=1)

    return df

train = create_features(train)
test = create_features(test)

# Drop early NaNs from train only
train = train.dropna().reset_index(drop=True)

# ============================================================
# 4. PREPARE MODEL MATRICES
# ============================================================

y = train["Target_Return"]

X = train.drop(columns=["Target_Return", "Time_Step"])
X_test = test.drop(columns=["Time_Step"])

# Align columns strictly
X_test = X_test[X.columns]

# Final NaN safety (important for Ridge)
X = X.fillna(0)
X_test = X_test.fillna(0)

# ============================================================
# 5. REGIME DETECTION (UNSUPERVISED)
# ============================================================

regime_cols = [c for c in X.columns if "roll_std20" in c]

kmeans = KMeans(n_clusters=3, random_state=42)
X["regime"] = kmeans.fit_predict(X[regime_cols])
X_test["regime"] = kmeans.predict(X_test[regime_cols])

# ============================================================
# 6. CUSTOM ASYMMETRIC OBJECTIVE (IAI ALIGNED)
# ============================================================

def asymmetric_objective(preds, train_data):
    labels = train_data.get_label()
    residual = preds - labels
    weights = np.where(labels < 0, 3.0, 1.0)
    grad = 2.0 * weights * residual
    hess = 2.0 * weights
    return grad, hess

# ============================================================
# 7. CUSTOM EVAL METRIC (WEIGHTED RMSE)
# ============================================================

def weighted_rmse_eval(preds, train_data):
    labels = train_data.get_label()
    weights = np.where(labels < 0, 3.0, 1.0)
    w_rmse = np.sqrt(np.sum(weights * (labels - preds)**2) / np.sum(weights))
    return "weighted_rmse", w_rmse, False  # lower is better

# ============================================================
# 8. IAI METRIC FOR LOCAL VALIDATION
# ============================================================

def iai_metric(y_true, y_pred):

    weights = np.where(y_true < 0, 3.0, 1.0)
    w_rmse = np.sqrt(np.sum(weights * (y_true - y_pred)**2) / np.sum(weights))
    score_downside = 1 / (1 + w_rmse)

    crash_threshold = np.percentile(y_true, 10)
    crash_mask = y_true <= crash_threshold

    rmse_crash = np.sqrt(mean_squared_error(y_true[crash_mask], y_pred[crash_mask]))
    score_tail = 1 / (1 + rmse_crash)

    corr, _ = spearmanr(y_true, y_pred)
    score_corr = max(0, corr)

    return 0.60*score_downside + 0.25*score_tail + 0.15*score_corr

# ============================================================
# 9. TIME SERIES CROSS VALIDATION
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val)

    params = {
        "learning_rate": 0.01,
        "num_leaves": 64,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "objective": asymmetric_objective,
        "metric": "None"
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        feval=weighted_rmse_eval,
        callbacks=[
            lgb.early_stopping(200),
            lgb.log_evaluation(100)
        ]
    )

    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 5

    fold_score = iai_metric(y_val.values, oof[val_idx])
    print(f"Fold {fold+1} IAI:", fold_score)

print("Overall CV IAI:", iai_metric(y.values, oof))

# ============================================================
# 10. RIDGE ENSEMBLE (RANK STABILIZER)
# ============================================================

ridge = Ridge(alpha=5.0)
ridge.fit(X, y)

ridge_test = ridge.predict(X_test)

# Blend
final_test_pred = 0.8 * test_preds + 0.2 * ridge_test
final_test_pred = np.clip(final_test_pred, -1, 1)

# ============================================================
# 11. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "Time_Step": test["Time_Step"],
    "Predicted_Return": final_test_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission file created successfully.")