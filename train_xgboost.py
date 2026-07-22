import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

INPUT_FILE = "nse_lstm_windows_vix.parquet"
MODEL_FILE = "xgb_stock_model.json"
LOOKBACK = 30
HORIZON = 5

BASE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "log_return", "rsi_14", "ema_12", "ema_26", "macd", "macd_signal",
    "sma_20", "sma_50", "dist_sma_20", "dist_sma_50",
    "roc_3", "roc_5", "roc_10", "bb_width", "atr_14", "atr_pct",
    "hist_vol_10", "hist_vol_21", "vol_zscore", "obv", "mfi_14",
    "body_size", "upper_shadow_ratio", "lower_shadow_ratio",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "bench_corr_20", "rel_strength",
    "vix_close", "vix_return", "vix_corr_20"
]

print("Loading dataset...")
df = pd.read_parquet(INPUT_FILE)

df["target_start"] = pd.to_datetime(df["target_start"])
# Split at mid 2025 to leave 6 months for validation
train_mask = df["target_start"] < "2025-06-30"
val_mask = df["target_start"] >= "2025-06-30"

# XGBoost doesn't need 3D windows. We just grab the absolute latest features
# available to the model on the day it makes the prediction (t30).
X_cols = [f"{feat}_t{LOOKBACK}" for feat in BASE_FEATURES]
y_cols = [f"y_t{h+1}" for h in range(HORIZON)]

print("Extracting features...")
X_train = df.loc[train_mask, X_cols].values
X_val = df.loc[val_mask, X_cols].values

# XGBoost natively predicts a single target. We want to predict the cumulative 
# 5-day return, so we simply sum the 5 horizon targets!
y_train = df.loc[train_mask, y_cols].sum(axis=1).values
y_val = df.loc[val_mask, y_cols].sum(axis=1).values

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

print("Training XGBoost Regressor...")
# Use relatively conservative hyperparameters to prevent overfitting on financial data
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=20,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=10
)

print(f"Saving model to {MODEL_FILE}")
model.save_model(MODEL_FILE)
print("Done!")
