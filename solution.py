import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

features = [c for c in train.columns if c.startswith("f")]

# ============================================================
# 2. MISSING VALUE HANDLING (NO LOOK-AHEAD)
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

    # Lags
    for lag in [1,2,3,5]:
        for f in features:
            df[f"{f}_lag{lag}"] = df[f].shift(lag)

    # Rolling volatility
    for f in features:
        df[f"{f}_roll_std5"] = df[f].rolling(5).std()
        df[f"{f}_roll_std20"] = df[f].rolling(20).std()

    # Rolling mean
    for f in features:
        df[f"{f}_roll_mean5"] = df[f].rolling(5).mean()

    # Cross interaction
    df["cross_vol"] = df[features].std(axis=1)
    df["cross_mean"] = df[features].mean(axis=1)

    return df

train = create_features(train)
test = create_features(test)

train = train.dropna().reset_index(drop=True)

X = train.drop(columns=["Target_Return"])
y = train["Target_Return"]

X_test = test.copy()

# ============================================================
# 4. REGIME DETECTION (UNSUPERVISED)
# ============================================================

regime_features = [c for c in X.columns if "roll_std20" in c]

kmeans = KMeans(n_clusters=3, random_state=42)
X["regime"] = kmeans.fit_predict(X[regime_features])
X_test["regime"] = kmeans.predict(X_test[regime_features].fillna(0))

# ============================================================
# 5. CUSTOM ASYMMETRIC LOSS FOR LIGHTGBM
# ============================================================

def asymmetric_obj(preds, train_data):
    labels = train_data.get_label()
    residual = preds - labels
    
    weights = np.where(labels < 0, 3.0, 1.0)
    
    grad = 2 * weights * residual
    hess = 2 * weights
    
    return grad, hess

# ============================================================
# 6. IAI METRIC IMPLEMENTATION (LOCAL VALIDATION)
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

    iai = 0.60*score_downside + 0.25*score_tail + 0.15*score_corr
    return iai

# ============================================================
# 7. TIME SERIES CROSS VALIDATION
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
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        fobj=asymmetric_obj,
        early_stopping_rounds=200,
        verbose_eval=False
    )

    oof[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / 5

    fold_score = iai_metric(y_val.values, oof[val_idx])
    print(f"Fold {fold+1} IAI:", fold_score)

print("Overall CV IAI:", iai_metric(y.values, oof))

# ============================================================
# 8. RANK-STABILIZING RIDGE ENSEMBLE
# ============================================================

ridge = Ridge(alpha=5.0)
ridge.fit(X, y)

ridge_test = ridge.predict(X_test)

# Blend
final_test_pred = 0.8 * test_preds + 0.2 * ridge_test

# Clip to required range
final_test_pred = np.clip(final_test_pred, -1, 1)

# ============================================================
# 9. CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "Time_Step": test["Time_Step"],
    "Predicted_Return": final_test_pred
})

submission.to_csv("submission.csv", index=False)

print("Submission file created.")